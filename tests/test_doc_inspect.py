"""Tests for app.doc_inspect — render/inspect/media-extract engine."""

from __future__ import annotations

import base64
import io
import zipfile
from pathlib import Path

import pytest

from app.doc_inspect import (
    DocInspectError,
    RenderResult,
    default_cache_dir,
    extract_embedded_media,
    inspect_document,
    parse_page_range,
    render_document,
)


# -- helpers -----------------------------------------------------------------


def _png_bytes(color=(200, 30, 30), size=(64, 48)) -> bytes:
    from PIL import Image

    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _make_pdf(path: Path, pages: int = 3) -> Path:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1} content")
    doc.save(str(path))
    doc.close()
    return path


def _make_pptx(path: Path, with_image: bool = False) -> Path:
    pptx = pytest.importorskip("pptx")
    from pptx.util import Inches

    prs = pptx.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    box = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(4), Inches(1))
    box.text_frame.text = "Hello Slide"
    if with_image:
        img_path = path.parent / "pic.png"
        img_path.write_bytes(_png_bytes())
        slide.shapes.add_picture(str(img_path), Inches(1), Inches(3), Inches(2), Inches(1))
    prs.save(str(path))
    return path


def _make_docx(path: Path) -> Path:
    docx = pytest.importorskip("docx")
    doc = docx.Document()
    doc.add_heading("Main Title", level=1)
    doc.add_paragraph("Some body text.")
    doc.add_table(rows=2, cols=3)
    doc.save(str(path))
    return path


def _make_xlsx(path: Path) -> Path:
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Data"
    ws["A1"] = 1
    ws["A2"] = "=A1*2"
    ws.merge_cells("B1:C1")
    wb.save(str(path))
    return path


# -- page range parser --------------------------------------------------------


def test_parse_page_range_default():
    assert parse_page_range(None, 10) == [1, 2, 3, 4]
    assert parse_page_range("", 2) == [1, 2]


def test_parse_page_range_explicit():
    assert parse_page_range("2", 10) == [2]
    assert parse_page_range("1-3", 10) == [1, 2, 3]
    assert parse_page_range("2,4,7", 10) == [2, 4, 7]
    assert parse_page_range("3-1", 10) == [1, 2, 3]


def test_parse_page_range_clamps_and_dedupes():
    assert parse_page_range("0", 5) == []
    assert parse_page_range("98-100", 5) == []
    assert parse_page_range("3-9", 5) == [3, 4, 5]
    assert parse_page_range("1,1,2-3", 5) == [1, 2, 3]
    assert parse_page_range("all", 3) == [1, 2, 3]
    assert parse_page_range("garbage", 5) == []


# -- render: images ------------------------------------------------------------


@pytest.mark.asyncio
async def test_render_plain_image(tmp_path):
    img = tmp_path / "photo.png"
    img.write_bytes(_png_bytes())
    res = await render_document(img, cache_dir=tmp_path / "cache")
    assert isinstance(res, RenderResult)
    assert res.total_pages == 1 and res.pages_rendered == [1]
    assert res.images[0].startswith("data:image/png;base64,")
    # decodable
    raw = base64.b64decode(res.images[0].split(",", 1)[1])
    assert raw[:4] == b"\x89PNG"


@pytest.mark.asyncio
async def test_render_image_downscales(tmp_path):
    from PIL import Image

    img = Image.new("RGB", (4000, 2000), (1, 2, 3))
    p = tmp_path / "big.png"
    img.save(p)
    res = await render_document(p, cache_dir=None, max_px=800)
    raw = base64.b64decode(res.images[0].split(",", 1)[1])
    out = Image.open(io.BytesIO(raw))
    assert max(out.size) <= 800


# -- render: pdf ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_render_pdf_pages(tmp_path):
    pdf = _make_pdf(tmp_path / "doc.pdf", pages=4)
    res = await render_document(pdf, pages="2-3", cache_dir=tmp_path / "cache")
    assert res.pages_rendered == [2, 3]
    assert res.total_pages == 4
    assert all(u.startswith("data:image/") for u in res.images)


@pytest.mark.asyncio
async def test_render_pdf_page_cap(tmp_path):
    pdf = _make_pdf(tmp_path / "doc.pdf", pages=10)
    res = await render_document(pdf, pages="all", max_images=3, cache_dir=None)
    assert len(res.images) == 3
    assert res.pages_rendered == [1, 2, 3]


@pytest.mark.asyncio
async def test_render_pdf_cache_hit(tmp_path):
    cache = tmp_path / "cache"
    pdf = _make_pdf(tmp_path / "doc.pdf", pages=2)
    first = await render_document(pdf, pages="1", cache_dir=cache)
    assert list((cache / "png").glob("*.img")), "page image should be cached"
    second = await render_document(pdf, pages="1", cache_dir=cache)
    assert second.images == first.images


@pytest.mark.asyncio
async def test_render_pdf_mtime_busts_cache(tmp_path):
    cache = tmp_path / "cache"
    pdf = _make_pdf(tmp_path / "doc.pdf", pages=2)
    await render_document(pdf, pages="1", cache_dir=cache)
    # modify file -> different key -> re-render (another cache entry appears)
    _make_pdf(tmp_path / "doc.pdf", pages=3)
    res = await render_document(pdf, pages="1", cache_dir=cache)
    assert res.total_pages == 3


@pytest.mark.asyncio
async def test_render_missing_and_unsupported(tmp_path):
    with pytest.raises(DocInspectError, match="not found"):
        await render_document(tmp_path / "nope.pdf", cache_dir=None)
    f = tmp_path / "x.bin"
    f.write_bytes(b"123")
    with pytest.raises(DocInspectError, match="unsupported"):
        await render_document(f, cache_dir=None)


@pytest.mark.asyncio
async def test_render_office_without_libreoffice(tmp_path):
    _make_pptx(tmp_path / "s.pptx")

    class NoLO:
        available = False

    with pytest.raises(DocInspectError, match="LibreOffice"):
        await render_document(tmp_path / "s.pptx", converter=NoLO(), cache_dir=None)


# -- inspect -------------------------------------------------------------------


def test_inspect_pptx_structure(tmp_path):
    _make_pptx(tmp_path / "deck.pptx", with_image=True)
    out = inspect_document(tmp_path / "deck.pptx")
    assert "PPTX: 1 slide" in out
    assert "Hello Slide" in out
    assert "cm" in out  # positions present
    assert "image" in out.lower()


def test_inspect_docx_structure(tmp_path):
    _make_docx(tmp_path / "doc.docx")
    out = inspect_document(tmp_path / "doc.docx")
    assert "DOCX:" in out
    assert "Main Title" in out
    assert "table" in out.lower()


def test_inspect_xlsx_structure(tmp_path):
    _make_xlsx(tmp_path / "book.xlsx")
    out = inspect_document(tmp_path / "book.xlsx")
    assert "XLSX" in out
    assert "Data" in out
    assert "merged" in out
    assert "formula" in out


def test_inspect_pdf(tmp_path):
    pdf = _make_pdf(tmp_path / "doc.pdf", pages=2)
    out = inspect_document(pdf)
    assert "PDF: 2 page(s)" in out
    assert "page 1:" in out


def test_inspect_image(tmp_path):
    p = tmp_path / "i.png"
    p.write_bytes(_png_bytes())
    out = inspect_document(p)
    assert "64x48" in out


def test_inspect_unsupported(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"123")
    with pytest.raises(DocInspectError):
        inspect_document(f)


# -- media extraction ----------------------------------------------------------


def test_extract_media_pptx(tmp_path):
    _make_pptx(tmp_path / "deck.pptx", with_image=True)
    media = extract_embedded_media(tmp_path / "deck.pptx")
    names = [n for n, _ in media]
    assert any(n.endswith(".png") for n in names)
    assert all(isinstance(d, bytes) and len(d) > 0 for _, d in media)


def test_extract_media_skips_embeddings(tmp_path):
    # hand-built zip: media ok, embeddings excluded
    f = tmp_path / "x.pptx"
    with zipfile.ZipFile(f, "w") as zf:
        zf.writestr("ppt/media/image1.png", _png_bytes())
        zf.writestr("ppt/embeddings/evil.bin", b"MZ..")
    media = extract_embedded_media(f)
    assert [n for n, _ in media] == ["image1.png"]


def test_extract_media_pdf(tmp_path):
    pytest.importorskip("fitz")
    pdf = _make_pdf(tmp_path / "doc.pdf", pages=1)
    # insert the image into the page
    img = tmp_path / "pic.png"
    img.write_bytes(_png_bytes())
    doc = __import__("fitz").open(str(pdf))
    import fitz as _fitz

    doc[0].insert_image(_fitz.Rect(10, 10, 50, 50), filename=str(img))
    doc.save(str(pdf), incremental=True, encryption=0)
    doc.close()
    media = extract_embedded_media(pdf)
    assert media and media[0][0].startswith("page1_img")


def test_extract_media_plain_file(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello")
    assert extract_embedded_media(f) == []


# -- cache dir -----------------------------------------------------------------


def test_default_cache_dir_is_sibling(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_WORKSPACE_RENDER_CACHE", raising=False)
    root = tmp_path / "data" / "agent_workspaces"
    cache = default_cache_dir(root)
    assert cache == tmp_path / "data" / "ws_render_cache"
    assert root not in cache.parents


def test_default_cache_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WORKSPACE_RENDER_CACHE", str(tmp_path / "custom"))
    assert default_cache_dir(tmp_path / "ws") == tmp_path / "custom"
