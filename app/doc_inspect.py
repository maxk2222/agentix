"""Visual and structural inspection of documents.

Used by the agent workspace tool so the agent can *see* binary documents
(PDF pages, rendered office files) and read their structure (slide shapes,
headings, tables, embedded media) instead of only decoded text.

All functions are filesystem-pure: callers are responsible for path
containment checks. Rendering is expensive (LibreOffice), so rendered
pages and converted PDFs are cached on disk under a caller-provided
cache directory.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import logging
import os
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

MAX_RENDER_IMAGES = 8
DEFAULT_RENDER_IMAGES = 4
DEFAULT_MAX_PX = 1568
_MAX_IMG_BYTES = 4 * 1024 * 1024
_JPEG_FALLBACK_BYTES = 3 * 1024 * 1024
_TOTAL_IMG_BYTES = 16 * 1024 * 1024
_CACHE_MAX_BYTES = 256 * 1024 * 1024
_CACHE_TTL_S = 24 * 3600
_INSPECT_SHAPE_CAP = 40
_INSPECT_PDF_PAGE_CAP = 50
_MEDIA_PAGE_CAP = 100
_MEDIA_MAX_COUNT = 200
_MEDIA_MAX_BYTES = 64 * 1024 * 1024
_XLSX_BIG_BYTES = 50 * 1024 * 1024

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
_LO_RENDER_EXTS = {
    ".pptx", ".ppt", ".docx", ".doc", ".xlsx", ".xls",
    ".pptm", ".docm", ".xlsm",
    ".odt", ".odp", ".ods", ".rtf",
}
_ZIP_MEDIA_PREFIXES = {
    ".pptx": "ppt/media/", ".pptm": "ppt/media/",
    ".docx": "word/media/", ".docm": "word/media/",
    ".xlsx": "xl/media/", ".xlsm": "xl/media/",
    ".odt": "Pictures/", ".odp": "Pictures/", ".ods": "Pictures/",
}
_MEDIA_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff",
    ".emf", ".wmf", ".svg", ".webp",
    ".mp3", ".mp4", ".wav", ".m4a", ".mov",
}

class DocInspectError(Exception):
    """Document cannot be rendered/inspected (missing dep, unsupported)."""


@dataclass
class RenderResult:
    images: List[str] = field(default_factory=list)
    pages_rendered: List[int] = field(default_factory=list)
    total_pages: int = 0
    source: str = ""


# ---------------------------------------------------------------------------
# page range parsing

def parse_page_range(spec: Optional[str], total: int, default_n: int = DEFAULT_RENDER_IMAGES) -> List[int]:
    """Parse ``"1-5"`` / ``"3"`` / ``"2,4,7"`` / ``"all"`` / None → 1-based page list."""
    if total < 1:
        return []
    if spec is None or not str(spec).strip():
        return list(range(1, min(total, default_n) + 1))
    spec = str(spec).strip().lower()
    if spec == "all":
        return list(range(1, total + 1))
    pages: set = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            try:
                lo, hi = int(a), int(b)
            except ValueError:
                continue
            if lo > hi:
                lo, hi = hi, lo
            # Clamp BEFORE materializing — "1-999999999" must not build a
            # billion-entry set for a 5-page document.
            lo = max(lo, 1)
            hi = min(hi, total)
            if lo <= hi:
                pages.update(range(lo, hi + 1))
        else:
            try:
                pages.add(int(part))
            except ValueError:
                continue
    return sorted(p for p in pages if 1 <= p <= total)


# ---------------------------------------------------------------------------
# caching

def _file_key(path: Path) -> str:
    st = path.stat()
    return hashlib.sha256(
        f"{path.resolve()}|{st.st_mtime_ns}|{st.st_size}".encode()
    ).hexdigest()[:32]


def _cache_get(cache_dir: Path, sub: str, key: str) -> Optional[Path]:
    if cache_dir is None:
        return None
    p = cache_dir / sub / f"{key}.{sub if sub != 'png' else 'img'}"
    try:
        if p.is_file():
            return p
    except OSError:
        return None
    return None


def _cache_put(cache_dir: Path, sub: str, key: str, data: bytes) -> Optional[Path]:
    if cache_dir is None:
        return None
    try:
        d = cache_dir / sub
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f".{key}.tmp"
        tmp.write_bytes(data)
        target = d / f"{key}.{sub if sub != 'png' else 'img'}"
        os.replace(tmp, target)
        return target
    except OSError as exc:
        logger.debug("render cache write failed: %s", exc)
        return None


def _evict_cache(cache_dir: Path) -> None:
    """Best-effort TTL + size-cap eviction."""
    if cache_dir is None or not cache_dir.is_dir():
        return
    try:
        now = time.time()
        entries = []
        total = 0
        for p in cache_dir.rglob("*"):
            if not p.is_file():
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            if now - st.st_mtime > _CACHE_TTL_S:
                p.unlink(missing_ok=True)
                continue
            entries.append((st.st_mtime, p))
            total += st.st_size
        if total > _CACHE_MAX_BYTES:
            entries.sort()
            for _, p in entries:
                try:
                    total -= p.stat().st_size
                    p.unlink()
                except OSError:
                    continue
                if total <= _CACHE_MAX_BYTES:
                    break
    except OSError as exc:
        logger.debug("render cache eviction failed: %s", exc)


# ---------------------------------------------------------------------------
# image encoding

def _data_uri(data: bytes) -> str:
    mime = "image/jpeg" if data[:2] == b"\xff\xd8" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _encode_image(img, max_bytes: int) -> Optional[bytes]:
    """Encode a PIL image ≤ max_bytes (PNG preferred, JPEG fallback, shrink last)."""
    from PIL import Image

    buf = io.BytesIO()
    img.save(buf, "PNG")
    data = buf.getvalue()
    if len(data) <= max_bytes and len(data) <= _JPEG_FALLBACK_BYTES:
        return data
    work = img.convert("RGB") if img.mode != "RGB" else img
    for _ in range(4):
        buf = io.BytesIO()
        work.save(buf, "JPEG", quality=85)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            return data
        w, h = work.size
        if w <= 256:
            break
        work = work.resize((max(256, w // 2), max(256, h // 2)), Image.LANCZOS)
    return data if len(data) <= max_bytes else None


def _open_scaled(path: Path, max_px: int):
    from PIL import Image

    img = Image.open(path)
    img.load()
    w, h = img.size
    if max(w, h) > max_px:
        ratio = max_px / max(w, h)
        img = img.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), Image.LANCZOS)
    return img


def _pixmap_to_pil(pix, max_px: int):
    from PIL import Image

    img = Image.open(io.BytesIO(pix.tobytes("png")))
    img.load()
    w, h = img.size
    if max(w, h) > max_px:
        ratio = max_px / max(w, h)
        img = img.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), Image.LANCZOS)
    return img


# ---------------------------------------------------------------------------
# rendering

async def render_document(
    path: Path,
    *,
    pages: Optional[str] = None,
    converter=None,
    cache_dir: Optional[Path] = None,
    max_images: int = DEFAULT_RENDER_IMAGES,
    max_px: int = DEFAULT_MAX_PX,
    max_bytes: int = _MAX_IMG_BYTES,
) -> RenderResult:
    """Render document pages/slides to image data URIs.

    ``path`` must already be existence- and containment-checked by the caller.
    ``converter`` is a ``LibreOfficeConverter`` (required for office formats).
    """
    path = Path(path)
    if not path.is_file():
        raise DocInspectError(f"file not found: {path.name}")
    ext = path.suffix.lower()
    max_images = max(1, min(int(max_images or DEFAULT_RENDER_IMAGES), MAX_RENDER_IMAGES))

    result: RenderResult
    if ext in _IMAGE_EXTS:
        data = await asyncio.to_thread(_render_image_file, path, max_px, max_bytes)
        if data is None:
            raise DocInspectError(f"image '{path.name}' exceeds size limits after downscale")
        result = RenderResult(images=[_data_uri(data)], pages_rendered=[1], total_pages=1, source="image")
    elif ext == ".pdf":
        result = await asyncio.to_thread(
            _render_pdf_file, path, pages, _file_key(path), cache_dir,
            max_images, max_px, max_bytes,
        )
    elif ext in _LO_RENDER_EXTS:
        if converter is None or not getattr(converter, "available", False):
            raise DocInspectError(
                f"rendering '{ext}' needs LibreOffice (soffice) which is not installed; "
                "use 'inspect' for structure-only info"
            )
        src_key = _file_key(path)
        cached_pdf = _cache_get(cache_dir, "pdf", src_key) if cache_dir else None
        if cached_pdf is not None:
            result = await asyncio.to_thread(
                _render_pdf_file, cached_pdf, pages, src_key, cache_dir,
                max_images, max_px, max_bytes,
            )
        else:
            with tempfile.TemporaryDirectory() as td:
                try:
                    out = await converter.convert(path, "pdf", Path(td))
                except Exception as exc:
                    raise DocInspectError(f"LibreOffice conversion failed for '{path.name}': {exc}") from exc
                pdf_path = out
                if cache_dir is not None:
                    stored = _cache_put(cache_dir, "pdf", src_key, out.read_bytes())
                    if stored is not None:
                        pdf_path = stored
                result = await asyncio.to_thread(
                    _render_pdf_file, pdf_path, pages, src_key, cache_dir,
                    max_images, max_px, max_bytes,
                )
    else:
        raise DocInspectError(f"unsupported format '{ext or path.name}' for view")
    # Eviction runs for every render — PDF/image-only workloads would
    # otherwise fill the cache unboundedly. Best-effort, rglob is cheap at
    # the 256 MB cap.
    _evict_cache(cache_dir)
    return result


def _render_image_file(path: Path, max_px: int, max_bytes: int) -> Optional[bytes]:
    try:
        img = _open_scaled(path, max_px)
    except Exception as exc:
        raise DocInspectError(f"cannot open image '{path.name}': {exc}") from exc
    return _encode_image(img, max_bytes)


def _render_pdf_file(
    pdf_path: Path,
    spec: Optional[str],
    src_key: str,
    cache_dir: Optional[Path],
    max_images: int,
    max_px: int,
    max_bytes: int,
) -> RenderResult:
    try:
        import fitz
    except ImportError as exc:
        raise DocInspectError("PyMuPDF (fitz) is not installed") from exc

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise DocInspectError(f"cannot open PDF for rendering: {exc}") from exc
    try:
        total = len(doc)
        wanted = parse_page_range(spec, total)[:max_images]
        if not wanted:
            raise DocInspectError(f"no pages in range (document has {total} pages)")
        result = RenderResult(total_pages=total, source="pdf")
        total_bytes = 0
        for page_no in wanted:
            data = None
            cached = _cache_get(cache_dir, "png", f"{src_key}_p{page_no}_{max_px}")
            if cached is not None:
                try:
                    data = cached.read_bytes()
                except OSError:
                    data = None
            if data is None:
                try:
                    pix = doc[page_no - 1].get_pixmap(matrix=fitz.Matrix(2, 2))
                except Exception as exc:
                    logger.warning("page %d render failed for %s: %s", page_no, pdf_path.name, exc)
                    continue
                img = _pixmap_to_pil(pix, max_px)
                data = _encode_image(img, max_bytes)
                if data is None:
                    logger.warning("page %d of %s exceeds size cap after downscale", page_no, pdf_path.name)
                    continue
                _cache_put(cache_dir, "png", f"{src_key}_p{page_no}_{max_px}", data)
            if total_bytes + len(data) > _TOTAL_IMG_BYTES:
                break
            total_bytes += len(data)
            result.images.append(_data_uri(data))
            result.pages_rendered.append(page_no)
        if not result.images:
            raise DocInspectError("no pages could be rendered within size limits")
        return result
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# structural inspection

def inspect_document(path: Path, *, converter=None) -> str:
    """Return a compact textual structure description for ``path``."""
    path = Path(path)
    if not path.is_file():
        raise DocInspectError(f"file not found: {path.name}")
    ext = path.suffix.lower()
    try:
        if ext in (".pptx", ".pptm"):
            return _inspect_pptx(path)
        if ext in (".docx", ".docm"):
            return _inspect_docx(path)
        if ext in (".xlsx", ".xlsm"):
            return _inspect_xlsx(path)
        if ext == ".pdf":
            return _inspect_pdf(path)
        if ext in _IMAGE_EXTS:
            return _inspect_image(path)
        if ext in _LO_RENDER_EXTS:
            return _inspect_legacy(path, converter)
    except DocInspectError:
        raise
    except Exception as exc:
        raise DocInspectError(f"inspect failed for '{path.name}': {exc}") from exc
    raise DocInspectError(f"unsupported format '{ext or path.name}' for inspect")


def _inspect_pptx(path: Path) -> str:
    try:
        from pptx import Presentation
    except ImportError as exc:
        raise DocInspectError("python-pptx is not installed") from exc
    prs = Presentation(str(path))
    cm = 360000.0
    lines = [
        f"PPTX: {len(prs.slides)} slide(s), "
        f"{prs.slide_width / cm:.1f}x{prs.slide_height / cm:.1f}cm"
    ]
    for i, slide in enumerate(prs.slides, 1):
        if i > _INSPECT_SHAPE_CAP:
            lines.append(f"... {len(prs.slides) - _INSPECT_SHAPE_CAP} more slide(s) omitted")
            break
        lines.append(f"Slide {i} — {len(slide.shapes)} shape(s):")
        for sh in slide.shapes:
            try:
                left = getattr(sh, "left", 0) / cm
                top = getattr(sh, "top", 0) / cm
                w = getattr(sh, "width", 0) / cm
                h = getattr(sh, "height", 0) / cm
                stype = str(getattr(sh, "shape_type", "?")).split(" ")[0]
                desc = f"  - {stype} '{sh.name}' @({left:.1f},{top:.1f}) {w:.1f}x{h:.1f}cm"
                if getattr(sh, "has_text_frame", False):
                    text = (sh.text_frame.text or "").strip().replace("\n", " / ")
                    if text:
                        desc += f' "{text[:100]}"'
                if getattr(sh, "has_chart", False):
                    desc += " [chart]"
                if getattr(sh, "has_table", False):
                    tbl = sh.table
                    desc += f" [table {len(tbl.rows)}x{len(tbl.columns)}]"
                if stype.startswith("PICTURE"):
                    try:
                        desc += f" [image:{sh.image.filename or sh.image.ext}]"
                    except Exception:
                        desc += " [image]"
                lines.append(desc)
            except Exception:
                lines.append("  - (unreadable shape)")
    return "\n".join(lines)


def _inspect_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise DocInspectError("python-docx is not installed") from exc
    doc = Document(str(path))
    paras = [p for p in doc.paragraphs if (p.text or "").strip()]
    lines = [
        f"DOCX: {len(doc.paragraphs)} paragraph(s) ({len(paras)} non-empty), "
        f"{len(doc.tables)} table(s), {len(doc.inline_shapes)} inline image(s), "
        f"{len(doc.sections)} section(s)"
    ]
    heads = [
        (p.style.name if p.style else "?", p.text.strip())
        for p in doc.paragraphs
        if p.style and str(p.style.name).lower().startswith(("heading", "berschrift", "title"))
    ]
    if heads:
        lines.append("Outline:")
        for name, text in heads[:_INSPECT_SHAPE_CAP]:
            lines.append(f"  {name}: {text[:100]}")
        if len(heads) > _INSPECT_SHAPE_CAP:
            lines.append(f"  ... {len(heads) - _INSPECT_SHAPE_CAP} more heading(s)")
    for idx, tbl in enumerate(doc.tables[:10], 1):
        lines.append(f"Table {idx}: {len(tbl.rows)} rows x {len(tbl.columns)} cols")
    if len(doc.tables) > 10:
        lines.append(f"... {len(doc.tables) - 10} more table(s)")
    return "\n".join(lines)


def _inspect_xlsx(path: Path) -> str:
    try:
        import openpyxl
    except ImportError as exc:
        raise DocInspectError("openpyxl is not installed") from exc
    # Very large workbooks go through read_only streaming — merged cells,
    # charts and embedded images are unavailable there but dims still work.
    big = path.stat().st_size > _XLSX_BIG_BYTES
    wb = openpyxl.load_workbook(str(path), read_only=big, data_only=False)
    lines = [f"XLSX: {len(wb.sheetnames)} sheet(s): {', '.join(wb.sheetnames[:20])}"]
    if big:
        lines.append("  (large file — streamed; merged cells/charts/images not reported)")
    for ws in wb.worksheets[:20]:
        merged = 0 if big else len(ws.merged_cells.ranges)
        formulas = 0
        cells = ws.max_row * ws.max_column if ws.max_row and ws.max_column else 0
        if not big and cells <= 200_000:
            for row in ws.iter_rows():
                for cell in row:
                    if cell.data_type == "f":
                        formulas += 1
                        if formulas > 10_000:
                            break
                if formulas > 10_000:
                    break
        charts = len(getattr(ws, "_charts", []) or [])
        imgs = len(getattr(ws, "_images", []) or [])
        parts = [f"  - {ws.title}: {ws.max_row}x{ws.max_column} ({ws.dimensions})"]
        if merged:
            parts.append(f"{merged} merged")
        if formulas:
            parts.append(f"{formulas}+ formulas")
        if charts:
            parts.append(f"{charts} chart(s)")
        if imgs:
            parts.append(f"{imgs} image(s)")
        lines.append(", ".join(parts))
    if len(wb.worksheets) > 20:
        lines.append(f"... {len(wb.worksheets) - 20} more sheet(s)")
    wb.close()
    return "\n".join(lines)


def _inspect_pdf(path: Path) -> str:
    try:
        import fitz
    except ImportError as exc:
        raise DocInspectError("PyMuPDF (fitz) is not installed") from exc
    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        raise DocInspectError(f"cannot open PDF: {exc}") from exc
    try:
        meta = doc.metadata or {}
        title = meta.get("title") or ""
        lines = [
            f"PDF: {len(doc)} page(s)"
            + (f", title '{title}'" if title else "")
            + (f", author '{meta.get('author')}'" if meta.get("author") else "")
        ]
        for i in range(min(len(doc), _INSPECT_PDF_PAGE_CAP)):
            page = doc[i]
            text_len = len(page.get_text().strip())
            img_count = len(page.get_images(full=True))
            lines.append(f"  page {i + 1}: {text_len} chars text, {img_count} image(s)")
        if len(doc) > _INSPECT_PDF_PAGE_CAP:
            lines.append(f"  ... {len(doc) - _INSPECT_PDF_PAGE_CAP} more page(s)")
        return "\n".join(lines)
    finally:
        doc.close()


def _inspect_image(path: Path) -> str:
    try:
        from PIL import Image
    except ImportError as exc:
        raise DocInspectError("Pillow is not installed") from exc
    with Image.open(path) as img:
        return f"Image: {img.format or '?'} {img.size[0]}x{img.size[1]}px, mode {img.mode}"


def _inspect_legacy(path: Path, converter) -> str:
    ext = path.suffix.lower()
    if converter is None or not getattr(converter, "available", False):
        return (
            f"Legacy office document '{path.name}' ({ext}): native structure extraction "
            "is not supported and LibreOffice is unavailable for conversion. "
            "Use 'view' to render pages when LibreOffice is installed."
        )
    raise DocInspectError(
        f"native structure extraction not supported for '{ext}'; "
        "use 'view' to render it (LibreOffice available)"
    )


# ---------------------------------------------------------------------------
# embedded media extraction

def extract_embedded_media(path: Path) -> List[Tuple[str, bytes]]:
    """Extract embedded media files (images/audio/video) from a document.

    Returns ``[(filename, bytes), ...]``. Office containers are ZIPs — media
    is read directly without executing anything. OLE ``embeddings/`` objects
    are deliberately excluded.
    """
    path = Path(path)
    if not path.is_file():
        raise DocInspectError(f"file not found: {path.name}")
    ext = path.suffix.lower()
    if ext in _ZIP_MEDIA_PREFIXES:
        return _zip_media(path, _ZIP_MEDIA_PREFIXES[ext])
    if ext == ".pdf":
        return _pdf_media(path)
    return []


def _zip_media(path: Path, prefix: str) -> List[Tuple[str, bytes]]:
    out: List[Tuple[str, bytes]] = []
    seen: set = set()
    try:
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if len(out) >= _MEDIA_MAX_COUNT:
                    break
                name = info.filename
                if name.endswith("/") or not name.startswith(prefix):
                    continue
                base = Path(name).name
                if Path(base).suffix.lower() not in _MEDIA_EXTS or base in seen:
                    continue
                # Unbounded zf.read() on a crafted container is a memory
                # bomb — skip oversized members instead.
                if info.file_size > _MEDIA_MAX_BYTES:
                    logger.debug("media member %s skipped (%d bytes)", name, info.file_size)
                    continue
                seen.add(base)
                try:
                    out.append((base, zf.read(name)))
                except Exception as exc:
                    logger.debug("media extract %s failed: %s", name, exc)
    except zipfile.BadZipFile as exc:
        raise DocInspectError(f"'{path.name}' is not a readable office container") from exc
    return out


def _pdf_media(path: Path) -> List[Tuple[str, bytes]]:
    try:
        import fitz
    except ImportError as exc:
        raise DocInspectError("PyMuPDF (fitz) is not installed") from exc
    out: List[Tuple[str, bytes]] = []
    seen_xref: set = set()
    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        raise DocInspectError(f"cannot open PDF: {exc}") from exc
    try:
        for pno in range(min(len(doc), _MEDIA_PAGE_CAP)):
            if len(out) >= _MEDIA_MAX_COUNT:
                break
            for img_idx, info in enumerate(doc[pno].get_images(full=True)):
                if len(out) >= _MEDIA_MAX_COUNT:
                    break
                xref = info[0]
                if xref in seen_xref:
                    continue
                seen_xref.add(xref)
                try:
                    extracted = doc.extract_image(xref)
                    data = extracted.get("image")
                    img_ext = extracted.get("ext", "png")
                    if data and len(data) <= _MEDIA_MAX_BYTES:
                        out.append((f"page{pno + 1}_img{img_idx + 1}.{img_ext}", data))
                except Exception as exc:
                    logger.debug("pdf image %d extract failed: %s", xref, exc)
        return out
    finally:
        doc.close()


def default_cache_dir(store_root: Path) -> Path:
    """Render cache next to (not inside) the workspace root so the orphan
    sweeper never treats it as an assistant directory."""
    override = os.environ.get("AGENT_WORKSPACE_RENDER_CACHE", "").strip()
    if override:
        return Path(override)
    return Path(store_root).parent / "ws_render_cache"
