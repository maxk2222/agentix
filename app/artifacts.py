"""Managed artifact storage and conversion for agent-generated files.

Artifacts (generated .docx/.pptx documents) live on disk under
``<data_dir>/artifacts/<conversation_id>/`` with a per-conversation
``_index.json`` recording filename -> {model, mime, size, kind, created_at}.
The stored ``model`` is needed for access checks on download because the
active-run registry expires shortly after a run finishes.

``LibreOfficeConverter`` wraps ``soffice --headless --convert-to`` (the same
approach the ingestion loader already uses) for previews and export formats;
it degrades gracefully when no LibreOffice binary is installed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_INDEX_FILE = "_index.json"
_CACHE_DIR = ".cache"
_MAX_FILE_BYTES = 25 * 1024 * 1024
_PREVIEW_DPI = 110
_PREVIEW_MAX_PAGES = 50
_CONVERT_TIMEOUT_S = 90

MIME_BY_EXT: Dict[str, str] = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".csv": "text/csv",
    ".pdf": "application/pdf",
    ".odt": "application/vnd.oasis.opendocument.text",
    ".odp": "application/vnd.oasis.opendocument.presentation",
    ".ods": "application/vnd.oasis.opendocument.spreadsheet",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".svg": "image/svg+xml",
    ".txt": "text/plain",
    ".log": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".json": "application/json",
    ".html": "text/html",
    ".htm": "text/html",
    ".xml": "application/xml",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".toml": "application/toml",
    ".py": "text/x-python",
    ".js": "text/javascript",
    ".css": "text/css",
    ".sh": "text/x-sh",
    ".sql": "application/sql",
    ".ini": "text/plain",
    ".cfg": "text/plain",
    ".env": "text/plain",
    ".rst": "text/x-rst",
    ".diff": "text/x-diff",
    ".patch": "text/x-diff",
}

# Which export formats each artifact kind can be converted to.
EXPORT_FORMATS: Dict[str, tuple] = {
    ".docx": ("pdf", "odt"),
    ".pptx": ("pdf", "odp"),
    ".xlsx": ("pdf", "ods"),
    ".csv": ("xlsx",),
}

# Extensions rasterized into a single preview page via Pillow.
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
# Extensions served as a raw text outline.
_TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".json", ".log", ".py", ".js", ".ts", ".jsx",
    ".tsx", ".css", ".scss", ".html", ".htm", ".xml", ".svg", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".sh", ".sql", ".env", ".rst", ".diff", ".patch",
}
# Longest side (px) of rasterized image previews.
_PREVIEW_MAX_DIM = 2048

# Index read-modify-write must not race across parallel tool calls. The lock
# registry is module-level because each tool (code_execution, agent_workspace,
# document_generator, the app itself) holds its own ArtifactStore instance.
_INDEX_LOCKS: Dict[Path, "asyncio.Lock"] = {}


class ArtifactError(RuntimeError):
    """Raised for expected, user-facing artifact failures."""


class ConversionError(ArtifactError):
    """LibreOffice conversion failed or is unavailable."""


def sanitize_filename(name: str) -> str:
    """Return a safe basename: no directory parts, no odd characters."""
    base = Path(name or "").name.strip()
    base = re.sub(r"[^\w.\-]", "_", base, flags=re.UNICODE)
    base = base.lstrip(".")
    return (base or "file")[:120]


def mime_for(name: str) -> str:
    return MIME_BY_EXT.get(Path(name).suffix.lower(), "application/octet-stream")


def kind_for(name: str, kind: str = "") -> str:
    """Stored ``kind`` for a file: explicit kinds win, generic/missing kinds
    (e.g. ``"file"`` from code_execution or workspace offers) derive from the
    extension so previews can key off it."""
    k = (kind or "").strip().lower()
    if k and k != "file":
        return k
    return Path(name).suffix.lower().lstrip(".") or "file"


class LibreOfficeConverter:
    """Serialize calls into a headless LibreOffice with an isolated profile."""

    def __init__(self, soffice_path: Optional[str] = None) -> None:
        self._soffice = soffice_path or self._detect()
        self._semaphore = asyncio.Semaphore(2)

    @staticmethod
    def _detect() -> Optional[str]:
        for env_key in ("LIBREOFFICE_PATH", "SOFFICE_PATH"):
            candidate = os.environ.get(env_key, "").strip()
            if candidate and Path(candidate).exists():
                return candidate
        for binary in ("soffice", "libreoffice"):
            found = shutil.which(binary)
            if found:
                return found
        mac_app = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
        if Path(mac_app).exists():
            return mac_app
        return None

    @property
    def available(self) -> bool:
        return self._soffice is not None

    async def convert(self, src: Path, fmt: str, outdir: Path) -> Path:
        """Convert ``src`` to ``fmt`` inside ``outdir``; return the output path."""
        if self._soffice is None:
            raise ConversionError("LibreOffice is not installed")
        src = Path(src)
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        expected = outdir / f"{src.stem}.{fmt}"
        async with self._semaphore:
            with tempfile.TemporaryDirectory() as profile_dir:
                try:
                    result = await asyncio.to_thread(
                        subprocess.run,
                        [
                            self._soffice,
                            "--headless",
                            "--norestore",
                            f"-env:UserInstallation=file://{profile_dir}",
                            "--convert-to",
                            fmt,
                            "--outdir",
                            str(outdir),
                            str(src),
                        ],
                        capture_output=True,
                        timeout=_CONVERT_TIMEOUT_S,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise ConversionError(f"LibreOffice conversion timed out after {_CONVERT_TIMEOUT_S}s") from exc
        if result.returncode != 0 or not expected.exists():
            stderr = result.stderr.decode(errors="replace")[:500] if result.stderr else ""
            raise ConversionError(f"LibreOffice conversion failed: {stderr or 'no output produced'}")
        return expected


class ArtifactStore:
    """Filesystem-backed artifact registry keyed by conversation."""

    def __init__(self, root: Path, converter: Optional[LibreOfficeConverter] = None) -> None:
        self._root = Path(root)
        self._converter = converter or LibreOfficeConverter()

    @property
    def converter(self) -> LibreOfficeConverter:
        return self._converter

    # -- paths --------------------------------------------------------------

    def _conv_dir(self, conversation_id: str) -> Path:
        return self._root / sanitize_filename(conversation_id)

    def _index_path(self, conversation_id: str) -> Path:
        return self._conv_dir(conversation_id) / _INDEX_FILE

    def _cache_dir(self, conversation_id: str, filename: str) -> Path:
        return self._conv_dir(conversation_id) / _CACHE_DIR / sanitize_filename(filename)

    def draft_dir(self, conversation_id: str) -> Path:
        """Directory for in-progress document_generator drafts.

        Drafts live under the conversation dir so delete_conversation() cleans
        them up; they are never registered in _index.json and therefore cannot
        be served through resolve()/page_path().
        """
        return self._conv_dir(conversation_id) / "_drafts"

    # -- index --------------------------------------------------------------

    def load_index(self, conversation_id: str) -> Dict[str, Any]:
        try:
            return json.loads(self._index_path(conversation_id).read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _index_lock(self, conversation_id: str) -> "asyncio.Lock":
        return _INDEX_LOCKS.setdefault(self._index_path(conversation_id), asyncio.Lock())

    def _save_index(self, conversation_id: str, index: Dict[str, Any]) -> None:
        # Atomic replace — a torn write would corrupt the whole index and
        # make every artifact of the conversation 404.
        path = self._index_path(conversation_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)

    def get_entry(self, conversation_id: str, filename: str) -> Optional[Dict[str, Any]]:
        safe = sanitize_filename(filename)
        entry = self.load_index(conversation_id).get(safe)
        if entry is None:
            return None
        # Derive the real kind for legacy entries stored as generic "file".
        return {"name": safe, **entry, "kind": kind_for(safe, entry.get("kind"))}

    # -- save / resolve ------------------------------------------------------

    async def save(
        self,
        conversation_id: str,
        model: str,
        filename: str,
        data: bytes,
        kind: str,
        *,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Write ``data`` as an artifact file; returns the stored entry.

        Default behaviour keeps every save unique (``name-2.ext`` on collision).
        With ``overwrite=True`` an existing entry of the same name is replaced
        in place — the index is updated and the preview cache invalidated so
        chat cards pointing at the name show the new version.
        """
        if len(data) > _MAX_FILE_BYTES:
            raise ArtifactError(f"Artifact exceeds {_MAX_FILE_BYTES // (1024 * 1024)} MB limit")
        conv_dir = self._conv_dir(conversation_id)
        conv_dir.mkdir(parents=True, exist_ok=True)

        # The whole load/dedupe/write/save-index sequence must stay atomic —
        # concurrent saves used to lose entries or corrupt the index, which
        # surfaced as cards that exist but 404 on download/preview.
        safe = sanitize_filename(filename)
        async with self._index_lock(conversation_id):
            index = self.load_index(conversation_id)
            if not overwrite:
                stem, suffix = Path(safe).stem, Path(safe).suffix
                candidate, counter = safe, 2
                while (conv_dir / candidate).exists() or candidate in index:
                    candidate = f"{stem}-{counter}{suffix}"
                    counter += 1
                safe = candidate

            path = conv_dir / safe
            await asyncio.to_thread(path.write_bytes, data)

            now = time.time()
            entry = index.get(safe) or {}
            kind = kind_for(safe, kind)
            entry.update({
                "model": model,
                "mime": mime_for(safe),
                "size": len(data),
                "kind": kind,
            })
            entry.setdefault("created_at", now)
            entry["updated_at"] = now
            index[safe] = entry
            self._save_index(conversation_id, index)

            if overwrite:
                cache = self._cache_dir(conversation_id, safe)
                if cache.exists():
                    await asyncio.to_thread(shutil.rmtree, cache, True)
        return {"name": safe, "conversation_id": conversation_id, "kind": kind,
                "mime": entry["mime"], "size": len(data), "updated_at": now}

    def resolve(self, conversation_id: str, filename: str) -> Optional[Path]:
        """Return the artifact path only if it is a registered index entry."""
        entry = self.get_entry(conversation_id, filename)
        if entry is None:
            return None
        conv_dir = self._conv_dir(conversation_id).resolve()
        path = (conv_dir / entry["name"]).resolve()
        if conv_dir not in path.parents or not path.is_file():
            return None
        return path

    async def delete_conversation(self, conversation_id: str) -> None:
        conv_dir = self._conv_dir(conversation_id)
        if conv_dir.exists():
            await asyncio.to_thread(shutil.rmtree, conv_dir, True)

    # -- preview -------------------------------------------------------------

    async def preview(self, conversation_id: str, filename: str) -> Dict[str, Any]:
        """Return preview payload: rendered page PNGs (PyMuPDF rasterizes PDFs
        directly, other office formats go through LibreOffice first), else a
        text outline."""
        entry = self.get_entry(conversation_id, filename)
        if entry is None:
            raise ArtifactError("artifact not found")
        path = self.resolve(conversation_id, filename)
        if path is None:
            raise ArtifactError("artifact file missing")
        base = {
            "kind": entry["kind"],
            "filename": entry["name"],
            "size": entry["size"],
            "mime": entry["mime"],
        }
        suffix = path.suffix.lower()
        if suffix in _TEXT_EXTS or (
            suffix != ".pdf" and suffix not in _IMAGE_EXTS and not self._converter.available
        ):
            base.update({"rendered": False, "outline": await asyncio.to_thread(self._outline, path)})
            return base
        try:
            pages = await self._render_pages(conversation_id, path)
        except Exception as exc:
            logger.warning("Artifact preview failed for %s: %s", path, exc)
            base.update({"rendered": False, "outline": await asyncio.to_thread(self._outline, path)})
            return base
        if not pages:
            logger.warning("Artifact preview rendered zero pages for %s", path)
            base.update({"rendered": False, "outline": await asyncio.to_thread(self._outline, path)})
            return base
        base.update({"rendered": True, "page_count": len(pages)})
        return base

    def page_path(self, conversation_id: str, filename: str, page: int) -> Optional[Path]:
        if page < 1:
            return None
        cache = self._cache_dir(conversation_id, filename)
        path = (cache / f"page_{page}.png").resolve()
        if cache.resolve() not in path.parents or not path.is_file():
            return None
        return path

    async def _render_pages(self, conversation_id: str, path: Path) -> List[Path]:
        """Rasterize artifact pages into the preview cache. PDFs go straight to
        PyMuPDF — LibreOffice is only needed to convert office formats first."""
        cache = self._cache_dir(conversation_id, path.name)
        existing = sorted(cache.glob("page_*.png")) if cache.exists() else []
        if existing:
            return existing
        suffix = path.suffix.lower()
        if suffix in _IMAGE_EXTS:
            cache.mkdir(parents=True, exist_ok=True)
            return await asyncio.to_thread(self._render_image_page, cache, path)
        if suffix == ".pdf":
            pdf = path
        else:
            if not self._converter.available:
                raise ConversionError("LibreOffice unavailable for preview conversion")
            pdf = await self._converter.convert(path, "pdf", cache)
        # Only ``convert()`` created the cache dir — the PDF path skipped it
        # and pix.save() then failed on the missing directory.
        cache.mkdir(parents=True, exist_ok=True)
        try:
            import pymupdf as fitz_mod  # PyMuPDF >= 1.26
        except ImportError:
            try:
                import fitz as fitz_mod  # older PyMuPDF releases
            except ImportError as exc:
                raise ConversionError("PyMuPDF not installed for preview rendering") from exc

        def _rasterize() -> List[Path]:
            pages: List[Path] = []
            try:
                doc = fitz_mod.open(str(pdf))
            except Exception as exc:
                raise ConversionError(f"cannot open preview PDF: {exc}") from exc
            try:
                for i, page in enumerate(doc):
                    if i >= _PREVIEW_MAX_PAGES:
                        break
                    pix = page.get_pixmap(dpi=_PREVIEW_DPI)
                    out = cache / f"page_{i + 1}.png"
                    pix.save(str(out))
                    pages.append(out)
            except Exception:
                # Drop partial pages — otherwise the next call would reuse the
                # truncated set from the ``existing`` glob and never retry.
                for partial in pages:
                    partial.unlink(missing_ok=True)
                raise
            finally:
                doc.close()
            return pages

        pages = await asyncio.to_thread(_rasterize)
        if not pages:
            raise ConversionError("preview produced zero pages")
        return pages

    @staticmethod
    def _render_image_page(cache: Path, path: Path) -> List[Path]:
        """Normalize any Pillow-readable image to a single PNG preview page."""
        try:
            from PIL import Image
        except ImportError as exc:
            raise ConversionError("Pillow not installed for image preview") from exc
        try:
            with Image.open(path) as src:
                if src.mode not in ("RGB", "RGBA", "L", "LA"):
                    img = src.convert("RGBA" if src.mode == "P" else "RGB")
                else:
                    img = src.copy()
                try:
                    if max(img.size) > _PREVIEW_MAX_DIM:
                        img.thumbnail((_PREVIEW_MAX_DIM, _PREVIEW_MAX_DIM), Image.LANCZOS)
                    out = cache / "page_1.png"
                    img.save(out, "PNG")
                finally:
                    img.close()
        except Exception as exc:
            raise ConversionError(f"cannot decode image for preview: {exc}") from exc
        return [out]

    def _outline(self, path: Path) -> List[Dict[str, Any]]:
        """Structured text fallback: sections for docx, slides for pptx,
        per-page text for pdf, one section per sheet for xlsx, raw lines
        for csv."""
        try:
            if path.suffix.lower() == ".pdf":
                try:
                    import pymupdf as fitz_mod
                except ImportError:
                    import fitz as fitz_mod
                doc = fitz_mod.open(str(path))
                pages = []
                try:
                    for i, pdf_page in enumerate(doc):
                        if i >= _PREVIEW_MAX_PAGES:
                            break
                        lines = [ln.strip() for ln in (pdf_page.get_text() or "").splitlines()]
                        lines = [ln for ln in lines if ln][:40]
                        pages.append({"title": f"{path.name} — Page {i + 1}", "lines": lines})
                finally:
                    doc.close()
                return pages
            if path.suffix.lower() == ".xlsx":
                from openpyxl import load_workbook
                # Values preferred (Excel caches them); files written by
                # openpyxl (document_generator) carry no cache, so formula
                # cells fall back to the formula text instead of blanks.
                wb_vals = load_workbook(str(path), read_only=True, data_only=True)
                wb_form = load_workbook(str(path), read_only=True, data_only=False)
                pages = []
                try:
                    for ws_v, ws_f in zip(wb_vals.worksheets, wb_form.worksheets):
                        lines: List[str] = []
                        for row_v, row_f in zip(
                            ws_v.iter_rows(max_row=12), ws_f.iter_rows(max_row=12)
                        ):
                            cells = []
                            for c_v, c_f in zip(row_v, row_f):
                                val = c_v.value if c_v.value is not None else c_f.value
                                cells.append("" if val is None else str(val))
                            while cells and not cells[-1]:
                                cells.pop()
                            line = "  |  ".join(cells)
                            if line.strip():
                                lines.append(line)
                        pages.append({"title": ws_v.title, "lines": lines})
                finally:
                    wb_vals.close()
                    wb_form.close()
                return pages[:_PREVIEW_MAX_PAGES]
            if path.suffix.lower() == ".csv":
                text = path.read_text(encoding="utf-8-sig", errors="replace")
                lines = [ln for ln in text.splitlines() if ln.strip()][:40]
                return [{"title": path.name, "lines": lines}]
            if path.suffix.lower() in _TEXT_EXTS:
                text = path.read_text(encoding="utf-8-sig", errors="replace")
                lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()][:40]
                return [{"title": path.name, "lines": lines}]
            if path.suffix.lower() == ".docx":
                from docx import Document
                doc = Document(str(path))
                pages: List[Dict[str, Any]] = []
                current: Dict[str, Any] = {"title": path.stem, "lines": []}
                for para in doc.paragraphs:
                    text = (para.text or "").strip()
                    if not text:
                        continue
                    style = (para.style.name if para.style else "") or ""
                    if style.lower().startswith("heading"):
                        if current["lines"]:
                            pages.append(current)
                        current = {"title": text, "lines": []}
                    else:
                        current["lines"].append(text)
                if current["lines"] or not pages:
                    pages.append(current)
                return pages[:_PREVIEW_MAX_PAGES]
            if path.suffix.lower() == ".pptx":
                from pptx import Presentation
                prs = Presentation(str(path))
                pages = []
                for i, slide in enumerate(prs.slides):
                    lines: List[str] = []
                    title = f"Slide {i + 1}"
                    for shape in slide.shapes:
                        if not shape.has_text_frame:
                            continue
                        for para in shape.text_frame.paragraphs:
                            text = "".join(run.text for run in para.runs).strip()
                            if not text:
                                continue
                            if shape is slide.shapes.title:
                                title = text
                            else:
                                lines.append(("  " * para.level) + text)
                    pages.append({"title": title, "lines": lines})
                return pages[:_PREVIEW_MAX_PAGES]
        except Exception as exc:
            logger.warning("Outline extraction failed for %s: %s", path, exc)
        return [{"title": path.name, "lines": []}]

    # -- export --------------------------------------------------------------

    async def export(self, conversation_id: str, filename: str, fmt: str) -> Path:
        """Return the converted file path for ``fmt`` (cached)."""
        fmt = fmt.lower().strip()
        path = self.resolve(conversation_id, filename)
        if path is None:
            raise ArtifactError("artifact not found")
        if fmt not in EXPORT_FORMATS.get(path.suffix.lower(), ()):
            raise ArtifactError(f"export format '{fmt}' not supported for '{path.suffix}'")
        cache = self._cache_dir(conversation_id, path.name)
        cached = cache / f"{path.stem}.{fmt}"
        if cached.exists():
            return cached
        return await self._converter.convert(path, fmt, cache)
