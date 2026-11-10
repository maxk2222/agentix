"""Conversation-scoped chat file attachments.

Uploaded documents (docx/xlsx/pptx/pdf/odt/csv/txt/md + legacy office) are
stored per conversation under ``data/chat_uploads/<conv_id>/<att_id>/`` —
never served back raw — and their extracted text is injected into the user
message on every ``/chat/completions`` turn as tagged ``<attachment>``
blocks. Images take the existing multimodal ``images`` path instead.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from app.artifacts import LibreOfficeConverter, sanitize_filename

logger = logging.getLogger(__name__)

_ORIGINAL_PREFIX = "original"
_META_FILE = "meta.json"
_TEXT_FILE = "extracted.txt"
_TEXT_EXTS = {".txt", ".md", ".csv"}
_LEGACY_MAP = {".doc": "docx", ".xls": "xlsx", ".ppt": "pptx"}
# Images go through KnowledgeLoader too — its image path runs OCR (tesseract)
# or LLM-vision extraction depending on the configured ocr_mode.
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
_LOADER_EXTS = {".docx", ".xlsx", ".pptx", ".pdf", ".odt"} | _IMAGE_EXTS
# Formats LLM providers accept as data-URI image_url parts — bmp/tiff stay
# on the OCR path only.
_VISION_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
_VISION_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


class AttachmentError(Exception):
    """Raised for validation/extraction failures surfaced to the client."""


class ChatAttachmentStore:
    """Filesystem store for user-uploaded conversation attachments."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def _conv_dir(self, conversation_id: str) -> Path:
        return self._root / sanitize_filename(conversation_id)

    def _att_dir(self, conversation_id: str, attachment_id: str) -> Path:
        conv_dir = self._conv_dir(conversation_id).resolve()
        path = (conv_dir / sanitize_filename(attachment_id)).resolve()
        if conv_dir not in path.parents:
            raise AttachmentError("Invalid attachment id.")
        return path

    # -- write / read -------------------------------------------------------

    async def save(self, conversation_id: str, filename: str, data: bytes) -> Dict[str, Any]:
        """Persist the raw upload; returns the attachment meta dict."""
        att_id = f"att-{uuid.uuid4().hex[:12]}"
        att_dir = self._att_dir(conversation_id, att_id)
        att_dir.mkdir(parents=True, exist_ok=True)
        ext = Path(filename or "").suffix.lower()
        original = att_dir / f"{_ORIGINAL_PREFIX}{ext}"
        tmp = att_dir / f".{_ORIGINAL_PREFIX}.tmp"
        await asyncio.to_thread(tmp.write_bytes, data)
        await asyncio.to_thread(os.replace, tmp, original)
        meta = {
            "id": att_id,
            "name": Path(filename or "file").name,
            "ext": ext,
            "size": len(data),
            "uploaded_at": time.time(),
            "char_count": 0,
            "truncated": False,
        }
        await self._write_meta(att_dir, meta)
        return meta

    async def set_extracted(self, conversation_id: str, attachment_id: str,
                            text: str, truncated: bool) -> Dict[str, Any]:
        att_dir = self._att_dir(conversation_id, attachment_id)
        meta = await asyncio.to_thread(self._read_meta, att_dir)
        if meta is None:
            raise AttachmentError("Attachment not found.")
        text_path = att_dir / _TEXT_FILE
        await asyncio.to_thread(text_path.write_text, text, "utf-8")
        meta["char_count"] = len(text)
        meta["truncated"] = truncated
        await self._write_meta(att_dir, meta)
        return meta

    def _read_meta(self, att_dir: Path) -> Optional[Dict[str, Any]]:
        try:
            return json.loads((att_dir / _META_FILE).read_text(encoding="utf-8"))
        except Exception:
            return None

    async def _write_meta(self, att_dir: Path, meta: Dict[str, Any]) -> None:
        meta_path = att_dir / _META_FILE
        tmp = att_dir / f".{_META_FILE}.tmp"
        await asyncio.to_thread(tmp.write_text, json.dumps(meta, ensure_ascii=False), "utf-8")
        await asyncio.to_thread(os.replace, tmp, meta_path)

    def list(self, conversation_id: str) -> List[Dict[str, Any]]:
        conv_dir = self._conv_dir(conversation_id)
        if not conv_dir.is_dir():
            return []
        items: List[Dict[str, Any]] = []
        for child in conv_dir.iterdir():
            if not child.is_dir():
                continue
            meta = self._read_meta(child)
            if meta is not None:
                items.append(meta)
        items.sort(key=lambda m: m.get("uploaded_at", 0.0))
        return items

    def get(self, conversation_id: str, attachment_id: str) -> Optional[Dict[str, Any]]:
        try:
            att_dir = self._att_dir(conversation_id, attachment_id)
        except AttachmentError:
            return None
        return self._read_meta(att_dir)

    def extracted_text(self, conversation_id: str, attachment_id: str) -> Optional[str]:
        try:
            att_dir = self._att_dir(conversation_id, attachment_id)
        except AttachmentError:
            return None
        try:
            return (att_dir / _TEXT_FILE).read_text(encoding="utf-8")
        except Exception:
            return None

    def original_path(self, conversation_id: str, attachment_id: str) -> Optional[Path]:
        try:
            att_dir = self._att_dir(conversation_id, attachment_id)
        except AttachmentError:
            return None
        for child in att_dir.iterdir():
            if child.name.startswith(_ORIGINAL_PREFIX) and child.is_file():
                return child
        return None

    async def delete(self, conversation_id: str, attachment_id: str) -> bool:
        try:
            att_dir = self._att_dir(conversation_id, attachment_id)
        except AttachmentError:
            return False
        if not att_dir.exists():
            return False
        await asyncio.to_thread(shutil.rmtree, att_dir, True)
        return True

    async def delete_conversation(self, conversation_id: str) -> None:
        conv_dir = self._conv_dir(conversation_id)
        if conv_dir.exists():
            await asyncio.to_thread(shutil.rmtree, conv_dir, True)


# -- extraction -------------------------------------------------------------

async def extract_attachment_text(
    path: Path,
    config: Any,
    *,
    llm_factory: Optional[Callable[[], Any]] = None,
) -> str:
    """Extract text content from an uploaded file.

    Plain text formats are read directly; modern office/PDF formats go through
    the existing ``KnowledgeLoader`` pipeline (UnstructuredLoader, optional PDF
    image / PPTX slide analysis via the configured flags); legacy office
    formats are converted via LibreOffice first.
    """
    path = Path(path)
    ext = path.suffix.lower()
    work_path = path
    tmp_dir: Optional[tempfile.TemporaryDirectory] = None
    try:
        if ext in _LEGACY_MAP:
            converter = LibreOfficeConverter()
            if not converter.available:
                raise AttachmentError("Legacy Office formats require LibreOffice on the server.")
            tmp_dir = tempfile.TemporaryDirectory()
            work_path = await converter.convert(path, _LEGACY_MAP[ext], Path(tmp_dir.name))
            ext = work_path.suffix.lower()

        if ext in _TEXT_EXTS:
            try:
                return work_path.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                raise AttachmentError(f"Could not read text file: {exc}") from exc

        if ext not in _LOADER_EXTS:
            raise AttachmentError(f"Unsupported file type '{ext}'.")

        from agent.loader import KnowledgeLoader

        loader = KnowledgeLoader(
            folder_path=work_path.parent,
            ocr_lang=getattr(config, "ocr_language", "deu"),
            tesseract_cmd=getattr(config, "tesseract_cmd", None),
            user_agent=getattr(config, "user_agent", None),
            max_concurrency=2,
            ocr_mode=getattr(config, "ocr_mode", "tesseract"),
            ocr_llm_prompt=getattr(config, "ocr_llm_prompt",
                                   "Extract all text and relevant information from this image."),
            ocr_llm_concurrency=2,
            llm_factory=llm_factory,
            pdf_image_extraction_enabled=getattr(config, "pdf_image_extraction_enabled", False),
            pdf_image_extraction_mode=getattr(config, "pdf_image_extraction_mode", "llm"),
            pptx_slide_analysis_enabled=getattr(config, "pptx_slide_analysis_enabled", False),
            pptx_slide_analysis_mode=getattr(config, "pptx_slide_analysis_mode", "llm"),
        )
        docs = await loader.load_local_documents(
            folder=work_path.parent, filepaths=[str(work_path)], file_timeout=None
        )
        text = "\n\n".join(d.page_content for d in docs if d.page_content)
        return text.strip()
    except AttachmentError:
        raise
    except Exception as exc:
        raise AttachmentError(f"Extraction failed: {exc}") from exc
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()


def build_attachment_context(
    store: ChatAttachmentStore,
    conversation_id: str,
    *,
    max_total_chars: int,
) -> str:
    """Assemble the ``<attachment>`` context block for a conversation.

    All stored attachments are included oldest-first, each already capped at
    extraction time; files that do not fit inside ``max_total_chars`` are
    represented by an ``omitted`` marker so the model knows they exist.
    """
    metas = store.list(conversation_id)
    if not metas:
        return ""
    blocks: List[str] = []
    used = 0
    for meta in metas:
        name = meta.get("name") or "file"
        ext = (meta.get("ext") or "").lstrip(".")
        text = store.extracted_text(conversation_id, meta["id"]) or ""
        block = f'<attachment name="{_xml_escape(name)}" type="{_xml_escape(ext)}">\n{text}\n</attachment>\n'
        if used + len(block) > max_total_chars:
            blocks.append(
                f'<attachment name="{_xml_escape(name)}" omitted="character-limit"/>\n'
            )
            continue
        blocks.append(block)
        used += len(block)
    return "\n" + "".join(blocks) if blocks else ""


async def image_attachment_data_uris(
    store: ChatAttachmentStore,
    conversation_id: str,
    metas: List[Dict[str, Any]],
    *,
    max_bytes: int,
    max_count: int,
    exclude_uris: Optional[set] = None,
) -> List[str]:
    """Return ``data:<mime>;base64,...`` URIs for stored image attachments.

    Newest-first (``metas`` arrives oldest-first), capped at ``max_count`` and
    ``max_bytes`` per file. Only provider-supported formats (``_VISION_EXTS``)
    qualify; missing/unreadable originals are skipped, never fatal.
    ``exclude_uris`` dedupes against inline request images.
    """
    import base64

    if max_count <= 0:
        return []
    seen = set(exclude_uris or ())
    uris: List[str] = []
    for meta in reversed(metas):
        if len(uris) >= max_count:
            break
        ext = (meta.get("ext") or "").lower()
        mime = _VISION_MIME.get(ext)
        if mime is None:
            continue
        path = store.original_path(conversation_id, meta.get("id") or "")
        if path is None:
            continue
        try:
            if path.stat().st_size > max_bytes:
                continue
            data = await asyncio.to_thread(path.read_bytes)
        except OSError as exc:
            logger.warning(
                "Skipping image attachment %s: unreadable original (%s)",
                meta.get("id"), exc,
            )
            continue
        uri = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        if uri in seen:
            continue
        seen.add(uri)
        uris.append(uri)
        meta["vision"] = True
    return uris


def _xml_escape(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
