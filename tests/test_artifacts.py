"""Tests for ArtifactStore and LibreOfficeConverter."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.artifacts import (
    ArtifactError,
    ArtifactStore,
    ConversionError,
    LibreOfficeConverter,
    sanitize_filename,
)


class _NoOffice(LibreOfficeConverter):
    def __init__(self):
        self._soffice = None
        import asyncio
        self._semaphore = asyncio.Semaphore(2)


@pytest.fixture()
def store(tmp_path):
    return ArtifactStore(tmp_path / "artifacts", converter=_NoOffice())


class TestFilenameSafety:
    def test_sanitize_strips_dirs_and_chars(self):
        assert sanitize_filename("../../etc/passwd") == "passwd"
        assert sanitize_filename("a/b/c.txt") == "c.txt"
        assert sanitize_filename("my file (1).pptx") == "my_file__1_.pptx"
        assert sanitize_filename("") == "file"


class TestStore:
    @pytest.mark.asyncio
    async def test_save_resolve_index_roundtrip(self, store):
        meta = await store.save("c1", "ModelA", "doc.pptx", b"data", "pptx")
        assert meta["name"] == "doc.pptx"
        assert store.resolve("c1", "doc.pptx") is not None
        entry = store.get_entry("c1", "doc.pptx")
        assert entry["model"] == "ModelA"
        assert entry["size"] == 4

    @pytest.mark.asyncio
    async def test_collision_gets_suffix(self, store):
        a = await store.save("c1", "M", "f.docx", b"1", "docx")
        b = await store.save("c1", "M", "f.docx", b"2", "docx")
        assert a["name"] != b["name"]

    @pytest.mark.asyncio
    async def test_resolve_rejects_unregistered(self, store, tmp_path):
        conv_dir = tmp_path / "artifacts" / "c1"
        conv_dir.mkdir(parents=True)
        (conv_dir / "evil.docx").write_bytes(b"x")
        assert store.resolve("c1", "evil.docx") is None  # file exists but not in index

    @pytest.mark.asyncio
    async def test_delete_conversation(self, store):
        await store.save("c1", "M", "f.docx", b"1", "docx")
        await store.delete_conversation("c1")
        assert store.load_index("c1") == {}

    @pytest.mark.asyncio
    async def test_size_cap(self, store):
        with pytest.raises(ArtifactError):
            await store.save("c1", "M", "big.docx", b"x" * (26 * 1024 * 1024), "docx")

    @pytest.mark.asyncio
    async def test_concurrent_saves_keep_all_entries(self, store):
        """Parallel tool calls share one index — no lost updates, no torn write."""
        names = [f"f{i}.txt" for i in range(8)]
        entries = await asyncio.gather(
            *(store.save("c1", "M", n, n.encode(), "file") for n in names)
        )
        for entry in entries:
            got = store.get_entry("c1", entry["name"])
            assert got is not None, entry["name"]
            assert store.resolve("c1", entry["name"]) is not None
        # Index must remain valid JSON containing every file.
        index = store.load_index("c1")
        assert len(index) == len(names)


class TestPreview:
    @pytest.mark.asyncio
    async def test_outline_fallback_without_soffice(self, store):
        from docx import Document
        doc = Document()
        doc.add_heading("Section A", level=1)
        doc.add_paragraph("body text")
        buf = Path("/tmp/x.docx")
        import io
        b = io.BytesIO()
        doc.save(b)
        await store.save("c1", "M", "rep.docx", b.getvalue(), "docx")
        preview = await store.preview("c1", "rep.docx")
        assert preview["rendered"] is False
        titles = [p["title"] for p in preview["outline"]]
        assert "Section A" in titles

    @pytest.mark.asyncio
    async def test_preview_missing_artifact(self, store):
        with pytest.raises(ArtifactError):
            await store.preview("c1", "nope.docx")

    @pytest.mark.asyncio
    async def test_pdf_renders_without_soffice(self, store):
        try:
            import pymupdf as fitz_mod
        except ImportError:
            fitz_mod = pytest.importorskip("fitz", reason="PyMuPDF not installed")
        doc = fitz_mod.open()
        pdf_page = doc.new_page()
        pdf_page.insert_text((72, 72), "hello preview")
        buf = doc.tobytes()
        doc.close()
        await store.save("c1", "M", "rep.pdf", buf, "pdf")
        preview = await store.preview("c1", "rep.pdf")
        assert preview["rendered"] is True
        assert preview["page_count"] == 1
        assert store.page_path("c1", "rep.pdf", 1) is not None

    @pytest.mark.asyncio
    async def test_pdf_outline_when_render_fails(self, store, monkeypatch):
        try:
            import pymupdf as fitz_mod
        except ImportError:
            fitz_mod = pytest.importorskip("fitz", reason="PyMuPDF not installed")
        doc = fitz_mod.open()
        pdf_page = doc.new_page()
        pdf_page.insert_text((72, 72), "fallback text line")
        buf = doc.tobytes()
        doc.close()
        await store.save("c1", "M", "rep.pdf", buf, "pdf")

        async def _boom(_conv, _path):
            raise ConversionError("no rasterizer")

        monkeypatch.setattr(store, "_render_pages", _boom)
        preview = await store.preview("c1", "rep.pdf")
        assert preview["rendered"] is False
        assert any("fallback text line" in ln for p in preview["outline"] for ln in p["lines"])

    @pytest.mark.asyncio
    async def test_render_zero_pages_falls_back(self, store, monkeypatch):
        await store.save("c1", "M", "rep.csv", b"a,b\n", "csv")

        async def _empty(_conv, _path):
            return []

        monkeypatch.setattr(store, "_render_pages", _empty)
        preview = await store.preview("c1", "rep.csv")
        assert preview["rendered"] is False
        assert preview["outline"]

    @pytest.mark.asyncio
    async def test_image_preview_renders_png(self, store, tmp_path):
        Image = pytest.importorskip("PIL.Image", reason="Pillow not installed")
        img = Image.new("RGB", (64, 32), (200, 30, 30))
        src = tmp_path / "plot.png"
        img.save(src)
        await store.save("c1", "M", "plot.png", src.read_bytes(), "file")
        preview = await store.preview("c1", "plot.png")
        assert preview["rendered"] is True
        assert preview["page_count"] == 1
        page = store.page_path("c1", "plot.png", 1)
        assert page is not None and page.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    @pytest.mark.asyncio
    async def test_text_outline_for_md(self, store):
        await store.save("c1", "M", "notes.md", b"# Head\n\nbody text\n", "file")
        preview = await store.preview("c1", "notes.md")
        assert preview["rendered"] is False
        lines = preview["outline"][0]["lines"]
        assert "# Head" in lines and "body text" in lines

    @pytest.mark.asyncio
    async def test_kind_derived_from_extension(self, store):
        await store.save("c1", "M", "chart.png", b"x", "file")
        entry = store.get_entry("c1", "chart.png")
        assert entry["kind"] == "png"
        await store.save("c1", "M", "doc.docx", b"x", "docx")
        assert store.get_entry("c1", "doc.docx")["kind"] == "docx"

    @pytest.mark.asyncio
    async def test_pdf_preview_creates_cache_dir(self, store, tmp_path):
        # Without the cache.mkdir the rasterizer failed on a missing parent —
        # the dir must exist even when rendering then fails (no fitz here).
        await store.save("c1", "M", "rep.pdf", b"%PDF-1.4 fake", "pdf")
        preview = await store.preview("c1", "rep.pdf")
        assert preview["rendered"] is False
        assert (tmp_path / "artifacts" / "c1" / ".cache" / "rep.pdf").is_dir()

    @pytest.mark.asyncio
    async def test_xlsx_outline_shows_formulas(self, store):
        import io as _io
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.append(["n", "doubled"])
        ws.append([2, "=A2*2"])
        buf = _io.BytesIO()
        wb.save(buf)
        await store.save("c1", "M", "calc.xlsx", buf.getvalue(), "xlsx")
        preview = await store.preview("c1", "calc.xlsx")
        assert preview["rendered"] is False
        assert any("=A2*2" in ln for p in preview["outline"] for ln in p["lines"])


class TestConverter:
    @pytest.mark.asyncio
    async def test_convert_invokes_soffice(self, tmp_path):
        converter = LibreOfficeConverter(soffice_path="/fake/soffice")
        src = tmp_path / "in.pptx"
        src.write_bytes(b"pptx")
        outdir = tmp_path / "out"

        def fake_run(cmd, **kwargs):
            outdir.mkdir(exist_ok=True)
            (outdir / "in.pdf").write_bytes(b"pdf")
            return MagicMock(returncode=0, stderr=b"")

        with patch("subprocess.run", side_effect=fake_run) as run:
            result = await converter.convert(src, "pdf", outdir)
        assert result.name == "in.pdf"
        assert "--headless" in run.call_args[0][0]

    @pytest.mark.asyncio
    async def test_convert_no_soffice(self):
        converter = _NoOffice()
        with pytest.raises(ConversionError):
            await converter.convert(Path("x.pptx"), "pdf", Path("/tmp"))

    @pytest.mark.asyncio
    async def test_export_format_gate(self, store):
        await store.save("c1", "M", "d.docx", b"x", "docx")
        with pytest.raises(ArtifactError):
            await store.export("c1", "d.docx", "odp")  # docx can't export odp


class TestDrafts:
    def test_draft_dir_under_conversation(self, store):
        d = store.draft_dir("c1")
        assert d.name == "_drafts"
        assert d.parent == store._conv_dir("c1")

    @pytest.mark.asyncio
    async def test_delete_conversation_removes_drafts(self, store):
        d = store.draft_dir("c1")
        d.mkdir(parents=True)
        (d / "doc-abc123.json").write_text("{}")
        await store.delete_conversation("c1")
        assert not d.exists()


class TestSpreadsheetArtifacts:
    def test_mime_types(self):
        from app.artifacts import mime_for
        assert "spreadsheetml" in mime_for("a.xlsx")
        assert mime_for("a.csv") == "text/csv"
        assert "opendocument.spreadsheet" in mime_for("a.ods")

    def test_export_formats(self):
        from app.artifacts import EXPORT_FORMATS
        assert EXPORT_FORMATS[".xlsx"] == ("pdf", "ods")
        assert EXPORT_FORMATS[".csv"] == ("xlsx",)

    @pytest.mark.asyncio
    async def test_xlsx_export_gate(self, store):
        await store.save("c1", "M", "t.csv", b"a,b", "csv")
        with pytest.raises(ArtifactError):
            await store.export("c1", "t.csv", "pdf")  # csv only exports xlsx

    @pytest.mark.asyncio
    async def test_xlsx_outline_fallback(self, store):
        import io as _io
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "Metrics"
        ws.append(["K", "V"])
        ws.append(["a", "1"])
        buf = _io.BytesIO()
        wb.save(buf)
        await store.save("c1", "M", "rep.xlsx", buf.getvalue(), "xlsx")
        preview = await store.preview("c1", "rep.xlsx")
        assert preview["rendered"] is False
        titles = [p["title"] for p in preview["outline"]]
        assert "Metrics" in titles
        assert any("a" in line for line in preview["outline"][0]["lines"])

    @pytest.mark.asyncio
    async def test_csv_outline_fallback(self, store):
        await store.save("c1", "M", "rep.csv", "a,b\nx,1\n".encode("utf-8-sig"), "csv")
        preview = await store.preview("c1", "rep.csv")
        assert preview["rendered"] is False
        assert preview["outline"][0]["lines"][0] == "a,b"

    @pytest.mark.asyncio
    async def test_overwrite_updates_in_place(self, store):
        a = await store.save("c1", "M", "f.docx", b"123", "docx")
        b = await store.save("c1", "M", "f.docx", b"123456", "docx", overwrite=True)
        assert b["name"] == "f.docx" == a["name"]  # no -2 suffix
        entry = store.get_entry("c1", "f.docx")
        assert entry["size"] == 6
        assert "updated_at" in entry
        assert "created_at" in entry
        index = store.load_index("c1")
        assert [k for k in index if k.startswith("f")] == ["f.docx"]

    @pytest.mark.asyncio
    async def test_overwrite_clears_preview_cache(self, store, tmp_path):
        meta = await store.save("c1", "M", "f.docx", b"1", "docx")
        cache = tmp_path / "artifacts" / "c1" / ".cache" / "f.docx"
        cache.mkdir(parents=True)
        (cache / "page_1.png").write_bytes(b"png")
        await store.save("c1", "M", "f.docx", b"2", "docx", overwrite=True)
        assert not cache.exists()

    @pytest.mark.asyncio
    async def test_overwrite_false_keeps_unique_names(self, store):
        a = await store.save("c1", "M", "f.docx", b"1", "docx")
        b = await store.save("c1", "M", "f.docx", b"2", "docx")
        assert b["name"] == "f-2.docx"
        assert a["name"] == "f.docx"
