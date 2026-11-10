"""Document loading utilities shared by the ingestion pipeline."""

import asyncio
import base64
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, List, Optional

import pytesseract
from PIL import Image
from langchain.schema import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_unstructured import UnstructuredLoader
import ipaddress
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class KnowledgeLoader:
    """Concurrent loader that merges local corpora, remote URLs, and OCR results.

    Image inputs are routed through Tesseract or LLM vision analysis (with
    Tesseract fallback), structured text files through
    ``langchain_unstructured``, and optional URLs through ``WebBaseLoader`` with
    a custom User-Agent when provided.
    """
    def __init__(
        self,
        folder_path: Path,
        urls: Optional[List[str]] = None,
        image_exts: Optional[List[str]] = None,
        text_exts: Optional[List[str]] = None,
        ocr_lang: str = "deu",
        tesseract_cmd: Optional[Path] = None,
        user_agent: Optional[str] = None,
        max_concurrency: int = 8,
        ocr_mode: str = "tesseract",
        ocr_llm_prompt: str = "Extract all text and relevant information from this image.",
        ocr_llm_concurrency: int = 4,
        llm_factory: Optional[Callable[[], Any]] = None,
        pdf_image_extraction_enabled: bool = False,
        pdf_image_extraction_mode: str = "llm",
        pptx_slide_analysis_enabled: bool = False,
        pptx_slide_analysis_mode: str = "llm",
    ) -> None:
        self.folder_path = folder_path if isinstance(folder_path, Path) else Path(folder_path)
        self.urls = urls or []
        self.image_exts = image_exts or [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"]
        self.text_exts = text_exts or [
            ".pdf", ".docx", ".txt", ".md",
            ".doc", ".odt",
            ".xls", ".xlsx",
            ".ppt", ".pptx",
            ".csv",
        ]
        self.ocr_lang = ocr_lang
        self._unstructured_languages = self._split_languages_for_unstructured(ocr_lang)
        self.user_agent = user_agent
        self._semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
        self._llm_semaphore = asyncio.Semaphore(max(1, int(ocr_llm_concurrency)))
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = str(tesseract_cmd)
        self.ocr_mode = ocr_mode
        self.ocr_llm_prompt = ocr_llm_prompt
        self.llm_factory = llm_factory
        self.pdf_image_extraction_enabled = pdf_image_extraction_enabled
        self.pdf_image_extraction_mode = pdf_image_extraction_mode
        self.pptx_slide_analysis_enabled = pptx_slide_analysis_enabled
        self.pptx_slide_analysis_mode = pptx_slide_analysis_mode

    async def load_documents(self) -> List[Document]:
        """Fetch remote and local sources concurrently and merge the results."""
        logger.info("Starting document loading process from '%s'", self.folder_path)
        web_docs_task = self.load_web_documents()
        local_docs_task = self.load_local_documents()
        web_docs, local_docs = await asyncio.gather(web_docs_task, local_docs_task)
        docs_list = web_docs + local_docs
        logger.info("Loaded %s documents in total", len(docs_list))
        return docs_list

    async def load_web_documents(self) -> List[Document]:
        """Fetch configured URLs while respecting a custom ``User-Agent`` when provided."""
        async def _is_safe_url(url: str) -> bool:
            try:
                parsed = urlparse(url)
                if parsed.scheme not in ("http", "https") or not parsed.netloc:
                    return False
                host = parsed.hostname
                if not host:
                    return False
                # if host is an IP, ensure it's not private/loopback
                try:
                    ip = ipaddress.ip_address(host)
                    return not (ip.is_private or ip.is_loopback or ip.is_reserved)
                except ValueError:
                    # host is a domain name; resolve asynchronously and check addresses
                    loop = asyncio.get_running_loop()
                    try:
                        infos = await loop.getaddrinfo(host, None)
                    except Exception:
                        return False
                    for family, _, _, _, sockaddr in infos:
                        addr = sockaddr[0]
                        try:
                            ip = ipaddress.ip_address(addr)
                            if ip.is_private or ip.is_loopback or ip.is_reserved:
                                return False
                        except Exception:
                            continue
                    return True
            except Exception:
                return False

        async def _load_one(url: str) -> List[Document]:
            if not await _is_safe_url(url):
                logger.warning("Skipping unsafe or unresolvable URL: %s", url)
                return []
            header_template = {"User-Agent": self.user_agent} if self.user_agent else {}
            loader_kwargs = {"header_template": header_template} if header_template else {}
            loader = WebBaseLoader(url, **loader_kwargs)
            async with self._semaphore:
                return await asyncio.to_thread(loader.load)

        if not self.urls:
            return []

        docs: List[Document] = []
        tasks = [_load_one(url) for url in self.urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for url, result in zip(self.urls, results):
            if isinstance(result, Exception):
                logger.warning("Error while loading URL %s: %s", url, result)
                continue
            docs.extend(result)
        return self._merge_documents_by_source(docs)

    async def load_local_documents(
        self,
        folder: Optional[Path] = None,
        filepaths: Optional[List[str]] = None,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
        file_timeout: Optional[float] = 900.0,
    ) -> List[Document]:
        """Walk the knowledge directory, enqueue OCR/text extraction tasks, and combine results.

        When ``filepaths`` is provided, only those files are processed instead of all files
        in the folder. ``progress_cb`` is invoked as ``(done, total, filepath)`` after each
        file finishes. ``file_timeout`` bounds a single file's processing so one pathological
        file cannot stall the whole batch (the abandoned worker keeps running in its thread,
        but the batch continues).
        """
        target_folder = folder or self.folder_path
        folder = target_folder if target_folder.is_absolute() else target_folder.resolve()
        if not folder.exists():
            logger.warning("The local path '%s' does not exist.", folder)
            return []

        if filepaths is not None:
            filepaths_to_process = filepaths
        else:
            filepaths_to_process = [str(p) for p in folder.rglob("*") if p.is_file()]

        if not filepaths_to_process:
            logger.info("No files found in the local directory.")
            return []

        total = len(filepaths_to_process)
        done = 0

        def _tick(path: str) -> None:
            nonlocal done
            done += 1
            if progress_cb is not None:
                try:
                    progress_cb(done, total, path)
                except Exception:
                    pass

        tasks: List[asyncio.Task] = []
        final_docs: List[Document] = []

        async def _run_limited(awaitable, filepath: str):
            async with self._semaphore:
                try:
                    if file_timeout and file_timeout > 0:
                        return await asyncio.wait_for(awaitable, timeout=file_timeout)
                    return await awaitable
                except asyncio.TimeoutError:
                    logger.warning(
                        "Timed out loading file '%s' after %.0fs; skipping.",
                        filepath, file_timeout or 0,
                    )
                    return []
                finally:
                    _tick(filepath)

        for filepath in filepaths_to_process:
            ext = os.path.splitext(filepath)[1].lower()
            if ext in self.image_exts:
                tasks.append(asyncio.create_task(_run_limited(
                    self.aextract_text_from_image(filepath, os.path.basename(filepath)), filepath
                )))
            elif ext in (".md", ".txt"):
                try:
                    text = Path(filepath).read_text(encoding="utf-8")
                    final_docs.append(
                        Document(page_content=text, metadata={"source": filepath})
                    )
                except Exception as e:
                    logger.warning(
                        "Error reading text file %s: %s", os.path.basename(filepath), e
                    )
                _tick(filepath)
            elif ext == ".pdf" and self.pdf_image_extraction_enabled:
                # Load text via UnstructuredLoader AND extract embedded images for LLM/Tesseract
                tasks.append(asyncio.create_task(_run_limited(
                    self._load_pdf_with_images(filepath), filepath
                )))
            elif ext == ".pptx" and self.pptx_slide_analysis_enabled:
                # Load text via UnstructuredLoader AND analyze slides as images
                tasks.append(asyncio.create_task(_run_limited(
                    self._load_pptx_with_slide_analysis(filepath), filepath
                )))
            elif ext in self.text_exts:
                try:
                    loader_kwargs = {}
                    if self._unstructured_languages:
                        loader_kwargs["languages"] = self._unstructured_languages
                    loader = UnstructuredLoader(filepath, **loader_kwargs)
                    tasks.append(asyncio.create_task(_run_limited(loader.aload(), filepath)))
                except Exception as e:
                    logger.warning(
                        "Error creating loading task for %s: %s", os.path.basename(filepath), e
                    )
                    _tick(filepath)
            else:
                _tick(filepath)

        if tasks:
            list_of_docs_lists = await asyncio.gather(*tasks, return_exceptions=True)
            for result in list_of_docs_lists:
                if isinstance(result, list):
                    final_docs.extend(result)
                elif isinstance(result, Exception):
                    logger.error("An error occurred during file loading: %s", result)

        return self._merge_documents_by_source(final_docs)


    async def aextract_text_from_image(self, image_path: str, source_filename: str) -> List[Document]:
        """Run OCR on an image asynchronously and wrap the result in a LangChain document.

        Uses LLM vision analysis, Tesseract, or both (auto mode) depending on ``ocr_mode``.
        """
        mode = self.ocr_mode
        text = ""

        if mode in ("llm", "auto"):
            llm_text = await self._extract_text_from_image_llm(image_path)
            if llm_text and llm_text.strip():
                text = llm_text
            elif mode == "llm":
                logger.warning("LLM vision returned empty result for %s", image_path)

        if not text.strip() and mode in ("tesseract", "auto"):
            loop = asyncio.get_running_loop()
            tess_text = await loop.run_in_executor(None, self._extract_text_from_image_sync, image_path)
            if tess_text and tess_text.strip():
                text = tess_text

        if text and text.strip():
            return [Document(page_content=text, metadata={"source": source_filename})]
        return []

    async def _extract_text_from_image_llm(self, image_path: str) -> str:
        """Use an LLM vision model to extract information from an image.

        Uses ``self.llm_factory`` to obtain a LangChain ``BaseChatModel`` instance.
        The image is sent as a Base64 data-URI in an OpenAI-compatible multimodal message.
        """
        if self.llm_factory is None:
            logger.debug("No LLM factory configured for image analysis, skipping LLM OCR")
            return ""

        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.error("Failed to read image %s for LLM analysis: %s", image_path, e)
            return ""

        # Determine MIME type from extension
        ext = os.path.splitext(image_path)[1].lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".tiff": "image/tiff",
            ".webp": "image/webp",
        }
        mime_type = mime_map.get(ext, "image/png")
        data_uri = f"data:{mime_type};base64,{image_data}"

        from langchain_core.messages import HumanMessage

        message = HumanMessage(content=[
            {"type": "text", "text": self.ocr_llm_prompt},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ])

        async with self._llm_semaphore:
            try:
                llm = self.llm_factory()
                if hasattr(llm, "ainvoke"):
                    result = await asyncio.wait_for(llm.ainvoke([message]), timeout=60.0)
                else:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(llm.invoke, [message]), timeout=60.0
                    )
                content = result.content if hasattr(result, "content") else str(result)
                return str(content).strip()
            except asyncio.TimeoutError:
                logger.warning("LLM vision analysis timed out for %s", image_path)
                return ""
            except Exception as e:
                logger.error("LLM vision analysis failed for %s: %s", image_path, e)
                return ""

    async def _analyze_image_bytes(self, image_bytes: bytes, mime_type: str, source_label: str) -> str:
        """Analyze raw image bytes using LLM or Tesseract based on the configured mode.

        Returns extracted text. Used by PDF image extraction and PPTX slide analysis.
        """
        import tempfile

        mode = self.pdf_image_extraction_mode  # reuse same mode logic
        text = ""

        if mode in ("llm", "auto") and self.llm_factory is not None:
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            data_uri = f"data:{mime_type};base64,{image_b64}"
            from langchain_core.messages import HumanMessage
            message = HumanMessage(content=[
                {"type": "text", "text": self.ocr_llm_prompt},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ])
            async with self._llm_semaphore:
                try:
                    llm = self.llm_factory()
                    if hasattr(llm, "ainvoke"):
                        result = await asyncio.wait_for(llm.ainvoke([message]), timeout=60.0)
                    else:
                        result = await asyncio.wait_for(
                            asyncio.to_thread(llm.invoke, [message]), timeout=60.0
                        )
                    text = str(result.content if hasattr(result, "content") else result).strip()
                except Exception as e:
                    logger.warning("LLM analysis failed for %s: %s", source_label, e)

        if not text.strip() and mode in ("tesseract", "auto"):
            # Write to temp file for Tesseract
            ext_map = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}
            tmp_ext = ext_map.get(mime_type, ".png")
            with tempfile.NamedTemporaryFile(suffix=tmp_ext, delete=False) as tmp:
                tmp.write(image_bytes)
                tmp_path = tmp.name
            try:
                loop = asyncio.get_running_loop()
                text = await loop.run_in_executor(None, self._extract_text_from_image_sync, tmp_path)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        return text

    async def _load_pdf_with_images(self, filepath: str) -> List[Document]:
        """Load a PDF: extract text via UnstructuredLoader and analyze embedded images.

        Uses PyMuPDF (fitz) to extract embedded images, then analyzes each image
        via LLM or Tesseract based on ``pdf_image_extraction_mode``.
        """
        source_name = os.path.basename(filepath)
        docs: List[Document] = []

        # 1. Load text content via UnstructuredLoader
        try:
            loader_kwargs = {}
            if self._unstructured_languages:
                loader_kwargs["languages"] = self._unstructured_languages
            loader = UnstructuredLoader(filepath, **loader_kwargs)
            text_docs = await loader.aload()
            docs.extend(text_docs)
        except Exception as e:
            logger.warning("UnstructuredLoader failed for %s: %s", source_name, e)

        # 2. Extract and analyze embedded images
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning("PyMuPDF (fitz) not installed, skipping PDF image extraction for %s", source_name)
            return docs

        try:
            pdf_doc = fitz.open(filepath)
            image_tasks: List[asyncio.Task] = []
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                for img_index, img_info in enumerate(page.get_images(full=True)):
                    xref = img_info[0]
                    try:
                        base_image = pdf_doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        mime_type = base_image.get("mime", "image/png")
                        label = f"{source_name}#page{page_num + 1}#img{img_index + 1}"
                        image_tasks.append(asyncio.create_task(
                            self._analyze_image_bytes(image_bytes, mime_type, label)
                        ))
                    except Exception as e:
                        logger.debug("Failed to extract image %d from page %d of %s: %s", img_index, page_num + 1, source_name, e)

            if image_tasks:
                results = await asyncio.gather(*image_tasks, return_exceptions=True)
                for idx, result in enumerate(results):
                    if isinstance(result, str) and result.strip():
                        docs.append(Document(
                            page_content=result,
                            metadata={"source": filepath, "type": "extracted_image", "image_index": idx},
                        ))
                    elif isinstance(result, Exception):
                        logger.warning("Image analysis failed in %s: %s", source_name, result)
            pdf_doc.close()
        except Exception as e:
            logger.warning("PDF image extraction failed for %s: %s", source_name, e)

        return docs

    async def _load_pptx_with_slide_analysis(self, filepath: str) -> List[Document]:
        """Load a PPTX: extract text via UnstructuredLoader and analyze slides as images.

        Uses python-pptx to extract text content and LibreOffice headless to render
        slides as images for LLM/Tesseract analysis.
        """
        source_name = os.path.basename(filepath)
        docs: List[Document] = []

        # 1. Load text content via UnstructuredLoader
        try:
            loader_kwargs = {}
            if self._unstructured_languages:
                loader_kwargs["languages"] = self._unstructured_languages
            loader = UnstructuredLoader(filepath, **loader_kwargs)
            text_docs = await loader.aload()
            docs.extend(text_docs)
        except Exception as e:
            logger.warning("UnstructuredLoader failed for %s: %s", source_name, e)

        # 2. Render slides as images via LibreOffice and analyze
        try:
            import subprocess
            import tempfile
            import glob as glob_mod

            with tempfile.TemporaryDirectory() as tmp_dir:
                # Convert PPTX to PDF using LibreOffice headless
                result = await asyncio.to_thread(
                    subprocess.run,
                    [
                        "libreoffice", "--headless", "--convert-to", "pdf",
                        "--outdir", tmp_dir, filepath,
                    ],
                    capture_output=True, timeout=120,
                )
                if result.returncode != 0:
                    logger.warning("LibreOffice conversion failed for %s: %s", source_name, result.stderr.decode()[:500])
                    return docs

                # Find the converted PDF
                pdf_files = glob_mod.glob(os.path.join(tmp_dir, "*.pdf"))
                if not pdf_files:
                    logger.warning("No PDF generated from PPTX %s", source_name)
                    return docs

                # Extract each page as an image using PyMuPDF
                try:
                    import fitz
                except ImportError:
                    logger.warning("PyMuPDF (fitz) not installed, cannot render PPTX slides for %s", source_name)
                    return docs

                pdf_doc = fitz.open(pdf_files[0])
                image_tasks: List[asyncio.Task] = []
                for page_num in range(len(pdf_doc)):
                    page = pdf_doc[page_num]
                    # Render page at 2x resolution for better OCR/vision quality
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    image_bytes = pix.tobytes("png")
                    label = f"{source_name}#slide{page_num + 1}"
                    image_tasks.append(asyncio.create_task(
                        self._analyze_image_bytes(image_bytes, "image/png", label)
                    ))

                if image_tasks:
                    results = await asyncio.gather(*image_tasks, return_exceptions=True)
                    for idx, result in enumerate(results):
                        if isinstance(result, str) and result.strip():
                            docs.append(Document(
                                page_content=result,
                                metadata={"source": filepath, "type": "slide_image", "slide_index": idx + 1},
                            ))
                        elif isinstance(result, Exception):
                            logger.warning("Slide analysis failed in %s: %s", source_name, result)
                pdf_doc.close()

        except FileNotFoundError:
            logger.warning("LibreOffice not found, skipping PPTX slide analysis for %s", source_name)
        except Exception as e:
            logger.warning("PPTX slide analysis failed for %s: %s", source_name, e)

        return docs

    def _extract_text_from_image_sync(self, image_path: str) -> str:
        """Perform blocking OCR extraction so it can be executed inside an executor."""
        try:
            image = Image.open(image_path)
            return pytesseract.image_to_string(image, lang=self.ocr_lang)
        except Exception as e:
            logger.error("Error processing image %s with Tesseract: %s", image_path, e)
            return ""

    @staticmethod
    def _split_languages_for_unstructured(lang_value: str | None) -> List[str]:
        """Convert Tesseract-style language strings (``deu+eng``) into lists."""
        if not lang_value:
            return []
        tokens = re.split(r"[+,]", lang_value)
        return [token.strip() for token in tokens if token.strip()]

    @staticmethod
    def _document_source_key(doc: Document, fallback_index: int) -> str:
        metadata = doc.metadata or {}
        for key in ("source_path", "source", "url", "file", "path", "filename"):
            value = metadata.get(key)
            if value:
                return str(value)
        return f"__document_{fallback_index}"

    @staticmethod
    def _document_page_number(doc: Document) -> Optional[object]:
        metadata = doc.metadata or {}
        page = metadata.get("page_number", metadata.get("page"))
        if page is None:
            return None
        if isinstance(page, str):
            page = page.strip()
        return page if page != "" else None

    def _merge_documents_by_source(self, docs: List[Document]) -> List[Document]:
        if not docs:
            return []

        grouped: dict[str, dict[str, Any]] = {}
        source_order: list[str] = []

        for index, doc in enumerate(docs):
            key = self._document_source_key(doc, index)
            if key not in grouped:
                grouped[key] = {
                    "metadata": dict(doc.metadata or {}),
                    "parts": [],
                    "page_numbers": [],
                }
                source_order.append(key)

            entry = grouped[key]
            content = (doc.page_content or "").strip()
            if content:
                entry["parts"].append(content)

            page_number = self._document_page_number(doc)
            if page_number is not None and page_number not in entry["page_numbers"]:
                entry["page_numbers"].append(page_number)

        merged_docs: List[Document] = []
        for key in source_order:
            entry = grouped[key]
            parts = entry["parts"]
            if not parts:
                continue

            metadata = dict(entry["metadata"])
            metadata.setdefault("source", key)
            if metadata.get("source_path") is None and key != metadata.get("source"):
                metadata["source_path"] = key
            metadata.pop("page", None)
            metadata.pop("page_number", None)

            page_numbers = entry["page_numbers"]
            if page_numbers:
                metadata["page_numbers"] = page_numbers
                metadata["page_count"] = len(page_numbers)

            metadata["element_count"] = len(parts)
            merged_docs.append(
                Document(
                    page_content="\n\n".join(parts),
                    metadata=metadata,
                )
            )

        logger.info("Merged %d raw documents into %d source documents.", len(docs), len(merged_docs))
        return merged_docs