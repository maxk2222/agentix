"""Tests for the document_generator builtin tool."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from agent.tools.builtin_tools.document_generator import DocumentGeneratorTool


@pytest.fixture()
def tool(tmp_path):
    return DocumentGeneratorTool(tmp_path / "artifacts")


CTX = {"conversation_id": "conv-1", "assistant_name": "AssistantX"}


class TestSchema:
    def test_tool_schema_surface(self, tool):
        ts = tool.tool_schema
        assert ts["function"]["name"] == "document_generator"
        params = ts["function"]["parameters"]
        assert params["required"] == []
        assert params["properties"]["kind"]["enum"] == ["docx", "pptx", "pdf", "xlsx", "csv"]
        assert params["properties"]["action"]["enum"] == [
            "create", "add_sections", "add_slides", "add_sheets", "edit", "finalize",
            "abort", "publish"]
        assert "draft_id" in params["properties"]
        assert "edits" in params["properties"]
        assert "sheets" in params["properties"]
        assert "deliver" in params["properties"]


class TestDocx:
    @pytest.mark.asyncio
    async def test_docx_build_and_reparse(self, tool):
        res = await tool.execute(
            kind="docx",
            title="Report",
            author="Author A",
            sections=[
                {"heading": "Intro", "level": 1, "paragraphs": ["Hello world."], "bullets": ["a", "b"]},
                {"heading": "Data", "table": {"headers": ["K", "V"], "rows": [["x", "1"]]},
                 "chart": {"type": "column", "categories": ["Q1"], "series": [{"name": "s", "values": [5]}]}},
            ],
            context=CTX,
        )
        assert res["ok"] is True
        file_meta = res["metadata"]["files"][0]
        assert file_meta["name"].endswith(".docx")
        assert file_meta["conversation_id"] == "conv-1"
        assert file_meta["kind"] == "docx"
        path = tool._store.resolve("conv-1", file_meta["name"])
        from docx import Document
        doc = Document(str(path))
        assert any("Hello world." in p.text for p in doc.paragraphs)
        assert len(doc.tables) == 1
        assert len(doc.inline_shapes) == 1  # matplotlib chart embedded


class TestPdf:
    @pytest.mark.asyncio
    async def test_pdf_kind_builds_via_converter(self, tool, monkeypatch):
        async def fake_convert(src, fmt, outdir):
            assert fmt == "pdf"
            out = Path(outdir) / "document.pdf"
            out.write_bytes(b"%PDF-1.4 fake")
            return out
        monkeypatch.setattr(tool._store.converter, "_soffice", "/fake/soffice")
        monkeypatch.setattr(tool._store.converter, "convert", fake_convert)
        res = await tool.execute(
            kind="pdf", title="Report",
            sections=[{"heading": "Intro", "paragraphs": ["Hello world."]}],
            context=CTX,
        )
        assert res["ok"] is True
        file_meta = res["metadata"]["files"][0]
        assert file_meta["name"].endswith(".pdf")
        assert file_meta["kind"] == "pdf"
        path = tool._store.resolve("conv-1", file_meta["name"])
        assert path.read_bytes() == b"%PDF-1.4 fake"

    @pytest.mark.asyncio
    async def test_pdf_kind_requires_libreoffice(self, tool, monkeypatch):
        monkeypatch.setattr(tool._store.converter, "_soffice", None)
        res = await tool.execute(
            kind="pdf", title="Report",
            sections=[{"heading": "Intro", "paragraphs": ["x"]}],
            context=CTX,
        )
        assert res["ok"] is False
        assert "LibreOffice" in res["error"]

    @pytest.mark.asyncio
    async def test_pdf_draft_flow(self, tool, monkeypatch):
        async def fake_convert(src, fmt, outdir):
            out = Path(outdir) / "document.pdf"
            out.write_bytes(b"%PDF-1.4 fake")
            return out
        monkeypatch.setattr(tool._store.converter, "_soffice", "/fake/soffice")
        monkeypatch.setattr(tool._store.converter, "convert", fake_convert)
        res = await tool.execute(action="create", kind="pdf", title="D", context=CTX)
        assert res["ok"] is True
        draft_id = res["content"].split("draft_id='")[1].split("'")[0]
        res = await tool.execute(action="add_sections", draft_id=draft_id,
                                 sections=[{"heading": "S", "paragraphs": ["p"]}], context=CTX)
        assert res["ok"] is True
        res = await tool.execute(action="add_slides", draft_id=draft_id,
                                 slides=[{"title": "s"}], context=CTX)
        assert res["ok"] is False  # pdf drafts are document drafts
        res = await tool.execute(action="finalize", draft_id=draft_id, context=CTX)
        assert res["ok"] is True
        assert res["metadata"]["files"][0]["name"].endswith(".pdf")


class TestPptx:
    @pytest.mark.asyncio
    async def test_pptx_build_with_native_chart(self, tool):
        res = await tool.execute(
            kind="pptx",
            title="Deck",
            slides=[
                {"layout": "title", "title": "Deck", "body": "Sub"},
                {"layout": "title_content", "title": "Agenda",
                 "bullets": ["One", {"text": "Sub", "level": 1}]},
                {"layout": "chart", "title": "Sales",
                 "chart": {"type": "column", "categories": ["Q1", "Q2"],
                           "series": [{"name": "a", "values": [1, 2]}, {"name": "b", "values": [2, 3]}]}},
                {"layout": "closing", "title": "End"},
            ],
            context=CTX,
        )
        assert res["ok"] is True
        meta = res["metadata"]["files"][0]
        path = tool._store.resolve("conv-1", meta["name"])
        from pptx import Presentation
        prs = Presentation(str(path))
        assert len(prs.slides) == 4
        charts = [sh for s in prs.slides for sh in s.shapes if sh.has_chart]
        assert len(charts) == 1  # native editable chart present

    @pytest.mark.asyncio
    async def test_pptx_image_chart_and_diagram(self, tool):
        res = await tool.execute(
            kind="pptx",
            title="Deck2",
            slides=[
                {"layout": "chart_bullets", "title": "Mix",
                 "chart": {"type": "pie", "render": "image", "categories": ["a", "b"],
                           "series": [{"name": "s", "values": [0.6, 0.4]}]},
                 "bullets": ["note"]},
                {"layout": "diagram", "title": "Flow",
                 "diagram": {"nodes": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
                             "edges": [["a", "b"]], "direction": "h"}},
            ],
            context=CTX,
        )
        assert res["ok"] is True
        path = tool._store.resolve("conv-1", res["metadata"]["files"][0]["name"])
        from pptx import Presentation
        prs = Presentation(str(path))
        assert len(prs.slides) == 2

    @pytest.mark.asyncio
    async def test_pptx_image_chart_no_graphviz_crash(self, tool, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _b: None)  # no graphviz → shape path
        res = await tool.execute(
            kind="pptx", title="D",
            slides=[{"layout": "diagram", "title": "F",
                     "diagram": {"nodes": [{"id": "a"}, {"id": "b"}], "edges": [["a", "b"]]}}],
            context=CTX,
        )
        assert res["ok"] is True


class TestValidationAndSafety:
    @pytest.mark.asyncio
    async def test_requires_conversation_id(self, tool):
        res = await tool.execute(kind="docx", title="T", sections=[{"paragraphs": ["x"]}])
        assert res["ok"] is False
        assert "conversation_id" in res["error"]

    @pytest.mark.asyncio
    async def test_bad_kind(self, tool):
        res = await tool.execute(kind="odt", title="T", context=CTX)
        assert res["ok"] is False

    @pytest.mark.asyncio
    async def test_empty_sections_rejected(self, tool):
        res = await tool.execute(kind="docx", title="T", sections=[], context=CTX)
        assert res["ok"] is False
        assert "sections" in res["error"]

    @pytest.mark.asyncio
    async def test_chart_requires_series(self, tool):
        res = await tool.execute(
            kind="pptx", title="T",
            slides=[{"layout": "chart", "title": "c", "chart": {"type": "pie", "categories": ["a"], "series": []}}],
            context=CTX,
        )
        assert res["ok"] is False
        assert "series" in res["error"]

    @pytest.mark.asyncio
    async def test_filename_traversal_sanitized(self, tool):
        res = await tool.execute(
            kind="docx", title="T", filename="../../etc/passwd.docx",
            sections=[{"paragraphs": ["x"]}], context=CTX,
        )
        assert res["ok"] is True
        name = res["metadata"]["files"][0]["name"]
        assert "/" not in name and ".." not in name

    @pytest.mark.asyncio
    async def test_filename_extension_forced(self, tool):
        res = await tool.execute(
            kind="docx", title="T", filename="out.txt",
            sections=[{"paragraphs": ["x"]}], context=CTX,
        )
        assert res["metadata"]["files"][0]["name"].endswith(".docx")

    @pytest.mark.asyncio
    async def test_theme_override_accepted(self, tool):
        res = await tool.execute(
            kind="docx", title="T", sections=[{"paragraphs": ["x"]}],
            context={**CTX, "tool_config": {"theme": {"accent": "#FF0000"}}},
        )
        assert res["ok"] is True


class TestArgNormalization:
    @pytest.mark.asyncio
    async def test_str_item_coerced(self, tool):
        res = await tool.execute(
            kind="docx", title="T",
            sections=[{"paragraphs": ["intro"]}, "a bare paragraph"],
            context=CTX,
        )
        assert res["ok"] is True
        path = tool._store.resolve("conv-1", res["metadata"]["files"][0]["name"])
        from docx import Document
        doc = Document(str(path))
        assert any("a bare paragraph" in p.text for p in doc.paragraphs)

    @pytest.mark.asyncio
    async def test_non_dict_item_indexed_error(self, tool):
        res = await tool.execute(
            kind="docx", title="T", sections=[{"paragraphs": ["x"]}, 42], context=CTX,
        )
        assert res["ok"] is False
        assert "sections[1]" in res["error"]

    @pytest.mark.asyncio
    async def test_inner_json_string_repair(self, tool):
        res = await tool.execute(
            kind="docx", title="T",
            sections=[{
                "paragraphs": ["x"],
                "table": '{"headers": ["a"], "rows": [["1"]]}',
                "chart": '{"type": "column", "categories": ["Q"], "series": [{"name": "s", "values": [1]}]}',
            }],
            context=CTX,
        )
        assert res["ok"] is True

    @pytest.mark.asyncio
    async def test_bad_inner_type_clean_error(self, tool):
        res = await tool.execute(
            kind="docx", title="T",
            sections=[{"paragraphs": ["x"], "table": 7}], context=CTX,
        )
        assert res["ok"] is False
        assert "table must be an object" in res["error"]

    @pytest.mark.asyncio
    async def test_slide_str_item_becomes_bullet(self, tool):
        res = await tool.execute(
            kind="pptx", title="T",
            slides=[{"layout": "title_content", "title": "S", "bullets": ["b"]}, "loose text"],
            context=CTX,
        )
        assert res["ok"] is True

    @pytest.mark.asyncio
    async def test_edges_dict_form_repaired(self, tool):
        res = await tool.execute(
            kind="pptx", title="T",
            slides=[{"layout": "diagram", "title": "F",
                     "diagram": {"nodes": [{"id": "a"}, {"id": "b"}],
                                 "edges": [{"from": "a", "to": "b"}]}}],
            context=CTX,
        )
        assert res["ok"] is True


class TestIncremental:
    @pytest.mark.asyncio
    async def test_create_add_finalize_docx(self, tool):
        res = await tool.execute(kind="docx", title="Draft", action="create", context=CTX)
        assert res["ok"] is True
        draft_id = res["content"].split("draft_id='")[1].split("'")[0]

        res = await tool.execute(action="add_sections", draft_id=draft_id,
                                 sections=[{"heading": "A", "paragraphs": ["1"]}], context=CTX)
        assert res["ok"] is True and "1/100" in res["content"]
        res = await tool.execute(action="add_sections", draft_id=draft_id,
                                 sections=[{"heading": "B", "paragraphs": ["2"]}], context=CTX)
        assert "2/100" in res["content"]

        res = await tool.execute(action="finalize", draft_id=draft_id, context=CTX)
        assert res["ok"] is True
        meta = res["metadata"]["files"][0]
        from docx import Document
        doc = Document(str(tool._store.resolve("conv-1", meta["name"])))
        texts = [p.text for p in doc.paragraphs]
        assert "A" in texts and "B" in texts
        # draft file removed on finalize
        assert not (tool._store.draft_dir("conv-1") / f"{draft_id}.json").exists()

    @pytest.mark.asyncio
    async def test_create_with_initial_content(self, tool):
        res = await tool.execute(
            kind="docx", title="D", action="create",
            sections=[{"heading": "S1", "paragraphs": ["x"]}], context=CTX)
        assert res["ok"] is True
        draft_id = res["content"].split("draft_id='")[1].split("'")[0]
        res = await tool.execute(action="finalize", draft_id=draft_id, context=CTX)
        assert res["ok"] is True

    @pytest.mark.asyncio
    async def test_incremental_pptx(self, tool):
        res = await tool.execute(kind="pptx", title="Deck", action="create", context=CTX)
        draft_id = res["content"].split("draft_id='")[1].split("'")[0]
        await tool.execute(action="add_slides", draft_id=draft_id,
                           slides=[{"layout": "title", "title": "Deck"}], context=CTX)
        res = await tool.execute(action="add_slides", draft_id=draft_id,
                                 slides=[{"layout": "closing", "title": "End"}], context=CTX)
        assert "2/60" in res["content"]
        res = await tool.execute(action="finalize", draft_id=draft_id, context=CTX)
        assert res["ok"] is True
        from pptx import Presentation
        prs = Presentation(str(tool._store.resolve("conv-1", res["metadata"]["files"][0]["name"])))
        assert len(prs.slides) == 2

    @pytest.mark.asyncio
    async def test_add_wrong_kind_rejected(self, tool):
        res = await tool.execute(kind="pptx", title="D", action="create", context=CTX)
        draft_id = res["content"].split("draft_id='")[1].split("'")[0]
        res = await tool.execute(action="add_sections", draft_id=draft_id,
                                 sections=[{"paragraphs": ["x"]}], context=CTX)
        assert res["ok"] is False
        assert "add_slides" in res["error"]

    @pytest.mark.asyncio
    async def test_unknown_draft(self, tool):
        res = await tool.execute(action="finalize", draft_id="doc-deadbeef99", context=CTX)
        assert res["ok"] is False
        assert "not found" in res["error"]

    @pytest.mark.asyncio
    async def test_malformed_draft_id(self, tool):
        res = await tool.execute(action="finalize", draft_id="../escape", context=CTX)
        assert res["ok"] is False

    @pytest.mark.asyncio
    async def test_abort_removes_draft(self, tool):
        res = await tool.execute(kind="docx", title="D", action="create", context=CTX)
        draft_id = res["content"].split("draft_id='")[1].split("'")[0]
        res = await tool.execute(action="abort", draft_id=draft_id, context=CTX)
        assert res["ok"] is True
        assert not (tool._store.draft_dir("conv-1") / f"{draft_id}.json").exists()

    @pytest.mark.asyncio
    async def test_cumulative_cap_error_keeps_draft(self, tool, monkeypatch):
        import agent.tools.builtin_tools.document_generator as dg
        monkeypatch.setattr(dg, "_MAX_SECTIONS", 2)
        res = await tool.execute(kind="docx", title="D", action="create", context=CTX)
        draft_id = res["content"].split("draft_id='")[1].split("'")[0]
        await tool.execute(action="add_sections", draft_id=draft_id,
                           sections=[{"paragraphs": ["a"]}, {"paragraphs": ["b"]}], context=CTX)
        res = await tool.execute(action="add_sections", draft_id=draft_id,
                                 sections=[{"paragraphs": ["c"]}], context=CTX)
        assert res["ok"] is False
        assert "too many sections" in res["error"]
        # draft still finalizable — oversized batch was rolled back
        res = await tool.execute(action="finalize", draft_id=draft_id, context=CTX)
        assert res["ok"] is True

    @pytest.mark.asyncio
    async def test_unknown_action(self, tool):
        res = await tool.execute(action="wat", kind="docx", title="T", context=CTX)
        assert res["ok"] is False
        assert "unknown action" in res["error"]


class TestDraftEdit:
    """action='edit' rewrites draft items via an ordered all-or-nothing batch."""

    async def _draft_id(self, tool, kind="docx", title="D"):
        res = await tool.execute(kind=kind, title=title, action="create", context=CTX)
        assert res["ok"] is True
        return res["content"].split("draft_id='")[1].split("'")[0]

    async def _draft_json(self, tool, draft_id):
        import json as _json
        path = tool._store.draft_dir("conv-1") / f"{draft_id}.json"
        return _json.loads(path.read_text())

    @pytest.mark.asyncio
    async def test_edit_all_op_types_sequential(self, tool):
        draft_id = await self._draft_id(tool)
        await tool.execute(action="add_sections", draft_id=draft_id,
                           sections=[{"heading": "A"}, {"heading": "B"}, {"heading": "C"}],
                           context=CTX)
        res = await tool.execute(
            action="edit", draft_id=draft_id,
            edits=[
                # [A,B,C] -> replace idx0 -> [A2,B,C]
                {"operation": "replace", "index": 0,
                 "section": {"heading": "A2", "paragraphs": ["new"]}},
                # -> insert at 2 -> [A2,B,X,C]
                {"operation": "insert", "index": 2, "section": {"heading": "X"}},
                # -> move 0->1 -> [B,A2,X,C]
                {"operation": "move", "from": 0, "to": 1},
                # -> remove last -> [B,A2,X]
                {"operation": "remove", "index": 3},
            ],
            context=CTX,
        )
        assert res["ok"] is True
        assert "Applied 4 edit(s)" in res["content"]
        assert "now has 3 sections" in res["content"]
        draft = await self._draft_json(tool, draft_id)
        assert [s.get("heading") for s in draft["sections"]] == ["B", "A2", "X"]
        assert draft["sections"][1]["paragraphs"] == ["new"]

    @pytest.mark.asyncio
    async def test_edit_then_finalize(self, tool):
        draft_id = await self._draft_id(tool, title="Final")
        await tool.execute(action="add_sections", draft_id=draft_id,
                           sections=[{"heading": "Keep"}, {"heading": "Drop"}], context=CTX)
        res = await tool.execute(action="edit", draft_id=draft_id,
                                 edits=[{"operation": "remove", "index": 1}], context=CTX)
        assert res["ok"] is True
        res = await tool.execute(action="finalize", draft_id=draft_id, context=CTX)
        assert res["ok"] is True
        from docx import Document
        doc = Document(str(tool._store.resolve("conv-1", res["metadata"]["files"][0]["name"])))
        texts = [p.text for p in doc.paragraphs]
        assert "Keep" in texts and "Drop" not in texts

    @pytest.mark.asyncio
    async def test_edit_failure_keeps_draft(self, tool):
        draft_id = await self._draft_id(tool)
        await tool.execute(action="add_sections", draft_id=draft_id,
                           sections=[{"heading": "A"}], context=CTX)
        res = await tool.execute(
            action="edit", draft_id=draft_id,
            edits=[
                {"operation": "replace", "index": 0, "section": {"heading": "A2"}},
                {"operation": "remove", "index": 5},  # out of bounds
            ],
            context=CTX,
        )
        assert res["ok"] is False
        assert "edit #1" in res["error"] and "draft unchanged" in res["error"]
        draft = await self._draft_json(tool, draft_id)
        assert [s.get("heading") for s in draft["sections"]] == ["A"]

    @pytest.mark.asyncio
    async def test_edit_pptx_uses_slide_payload(self, tool):
        draft_id = await self._draft_id(tool, kind="pptx", title="Deck")
        await tool.execute(action="add_slides", draft_id=draft_id,
                           slides=[{"layout": "title", "title": "One"}], context=CTX)
        res = await tool.execute(
            action="edit", draft_id=draft_id,
            edits=[{"operation": "replace", "index": 0,
                    "slide": {"layout": "closing", "title": "End"}}],
            context=CTX,
        )
        assert res["ok"] is True
        assert "now has 1 slides" in res["content"]
        draft = await self._draft_json(tool, draft_id)
        assert draft["slides"][0]["layout"] == "closing"

    @pytest.mark.asyncio
    async def test_edit_wrong_payload_key(self, tool):
        draft_id = await self._draft_id(tool, kind="pptx", title="Deck")
        await tool.execute(action="add_slides", draft_id=draft_id,
                           slides=[{"layout": "title"}], context=CTX)
        res = await tool.execute(
            action="edit", draft_id=draft_id,
            edits=[{"operation": "replace", "index": 0,
                    "section": {"heading": "wrong key"}}],
            context=CTX,
        )
        assert res["ok"] is False
        assert "'slide'" in res["error"]

    @pytest.mark.asyncio
    async def test_edit_empty_and_oversized(self, tool):
        draft_id = await self._draft_id(tool)
        res = await tool.execute(action="edit", draft_id=draft_id, edits=[], context=CTX)
        assert res["ok"] is False and "non-empty" in res["error"]
        res = await tool.execute(
            action="edit", draft_id=draft_id,
            edits=[{"operation": "remove", "index": 0}] * 51, context=CTX)
        assert res["ok"] is False and "max 50" in res["error"]

    @pytest.mark.asyncio
    async def test_edit_stringified_index(self, tool):
        draft_id = await self._draft_id(tool)
        await tool.execute(action="add_sections", draft_id=draft_id,
                           sections=[{"heading": "A"}, {"heading": "B"}], context=CTX)
        res = await tool.execute(
            action="edit", draft_id=draft_id,
            edits=[{"operation": "remove", "index": "0"}], context=CTX)
        assert res["ok"] is True
        draft = await self._draft_json(tool, draft_id)
        assert [s.get("heading") for s in draft["sections"]] == ["B"]

    @pytest.mark.asyncio
    async def test_edit_unknown_operation(self, tool):
        draft_id = await self._draft_id(tool)
        res = await tool.execute(
            action="edit", draft_id=draft_id,
            edits=[{"operation": "frobnicate", "index": 0}], context=CTX)
        assert res["ok"] is False and "unknown operation" in res["error"]

    @pytest.mark.asyncio
    async def test_edit_move_noop(self, tool):
        draft_id = await self._draft_id(tool)
        await tool.execute(action="add_sections", draft_id=draft_id,
                           sections=[{"heading": "A"}, {"heading": "B"}], context=CTX)
        res = await tool.execute(
            action="edit", draft_id=draft_id,
            edits=[{"operation": "move", "from": 0, "to": 0}], context=CTX)
        assert res["ok"] is True and "move 0→0" in res["content"]


class TestXlsx:
    @pytest.mark.asyncio
    async def test_xlsx_build_and_reparse(self, tool):
        res = await tool.execute(
            kind="xlsx", title="Report",
            sheets=[{
                "name": "Sales",
                "headers": ["Month", "Revenue", "Note"],
                "rows": [["Jan", 1200.5, "ok"], ["Feb", 980, None]],
                "freeze_header": True,
                "auto_filter": True,
            }],
            context=CTX,
        )
        assert res["ok"] is True
        file_meta = res["metadata"]["files"][0]
        assert file_meta["name"].endswith(".xlsx")
        assert file_meta["kind"] == "xlsx"
        path = tool._store.resolve("conv-1", file_meta["name"])
        from openpyxl import load_workbook
        wb = load_workbook(str(path))
        ws = wb["Sales"]
        assert ws["A1"].value == "Month" and ws["A1"].font.bold
        assert ws["B2"].value == 1200.5  # numeric typed
        assert ws["C3"].value is None    # None stays empty
        assert ws.freeze_panes == "A2"
        assert ws.auto_filter.ref is not None
        wb.close()

    @pytest.mark.asyncio
    async def test_xlsx_cell_types(self, tool):
        res = await tool.execute(
            kind="xlsx", title="T",
            sheets=[{
                "name": "Data",
                "headers": ["n", "b", "d", "f"],
                "rows": [[42, True, "2024-03-15", "=SUM(A2:A2)"]],
            }],
            context=CTX,
        )
        assert res["ok"] is True
        path = tool._store.resolve("conv-1", res["metadata"]["files"][0]["name"])
        from openpyxl import load_workbook
        wb = load_workbook(str(path))
        ws = wb.active
        assert ws["A2"].value == 42
        assert ws["B2"].value is True
        assert ws["C2"].is_date
        assert ws["D2"].value == "=SUM(A2:A2)"
        wb.close()

    @pytest.mark.asyncio
    async def test_xlsx_multi_sheet_names_sanitized(self, tool):
        res = await tool.execute(
            kind="xlsx", title="T",
            sheets=[
                {"name": "A/B:C", "headers": ["x"], "rows": [[1]]},
                {"name": "A/B:C", "headers": ["y"], "rows": [[2]]},
                {"name": "", "headers": ["z"], "rows": [[3]]},
            ],
            context=CTX,
        )
        assert res["ok"] is True
        path = tool._store.resolve("conv-1", res["metadata"]["files"][0]["name"])
        from openpyxl import load_workbook
        wb = load_workbook(str(path))
        assert len(wb.sheetnames) == 3
        assert len(set(wb.sheetnames)) == 3
        assert all(len(n) <= 31 and "/" not in n and ":" not in n for n in wb.sheetnames)
        wb.close()

    @pytest.mark.asyncio
    async def test_xlsx_chart_hidden_datasheet(self, tool):
        res = await tool.execute(
            kind="xlsx", title="T",
            sheets=[{
                "name": "Data",
                "headers": ["M", "V"],
                "rows": [["Jan", 5], ["Feb", 7]],
                "chart": {"type": "column", "title": "Rev",
                          "categories": ["Jan", "Feb"],
                          "series": [{"name": "Rev", "values": [5, 7]}]},
            }],
            context=CTX,
        )
        assert res["ok"] is True
        path = tool._store.resolve("conv-1", res["metadata"]["files"][0]["name"])
        from openpyxl import load_workbook
        wb = load_workbook(str(path))
        ws = wb["Data"]
        assert len(ws._charts) == 1
        assert "_chartdata" in wb.sheetnames
        assert wb["_chartdata"].sheet_state == "hidden"
        wb.close()

    @pytest.mark.asyncio
    async def test_xlsx_column_widths_and_formats(self, tool):
        res = await tool.execute(
            kind="xlsx", title="T",
            sheets=[{
                "name": "S",
                "headers": ["A", "B"],
                "rows": [[1.234, 0.5]],
                "column_widths": {"A": 30},
                "number_formats": {"B": "0.0%"},
            }],
            context=CTX,
        )
        assert res["ok"] is True
        path = tool._store.resolve("conv-1", res["metadata"]["files"][0]["name"])
        from openpyxl import load_workbook
        wb = load_workbook(str(path))
        ws = wb.active
        assert ws.column_dimensions["A"].width == 30.0
        assert ws["B2"].number_format == "0.0%"
        wb.close()


class TestCsv:
    @pytest.mark.asyncio
    async def test_csv_first_sheet_only(self, tool):
        res = await tool.execute(
            kind="csv", title="T",
            sheets=[
                {"name": "One", "headers": ["a", "b"], "rows": [["x", "1"], ["y", "2"]]},
                {"name": "Two", "headers": ["c"], "rows": [["z"]]},
            ],
            context=CTX,
        )
        assert res["ok"] is True
        file_meta = res["metadata"]["files"][0]
        assert file_meta["name"].endswith(".csv")
        assert file_meta["kind"] == "csv"
        path = tool._store.resolve("conv-1", file_meta["name"])
        text = path.read_bytes().decode("utf-8-sig")
        assert "x,1" in text.replace('"', "")
        assert "z" not in text.splitlines()[0]  # second sheet ignored

    @pytest.mark.asyncio
    async def test_csv_extension_forced(self, tool):
        res = await tool.execute(
            kind="csv", title="T", filename="out.txt",
            sheets=[{"headers": ["a"], "rows": [["1"]]}],
            context=CTX,
        )
        assert res["ok"] is True
        assert res["metadata"]["files"][0]["name"].endswith(".csv")

    @pytest.mark.asyncio
    async def test_csv_no_libreoffice_needed(self, tool, monkeypatch):
        monkeypatch.setattr(tool._store.converter, "_soffice", None)
        res = await tool.execute(
            kind="csv", title="T",
            sheets=[{"headers": ["a"], "rows": [["1"]]}],
            context=CTX,
        )
        assert res["ok"] is True


class TestSheetsValidation:
    @pytest.mark.asyncio
    async def test_empty_sheets_rejected(self, tool):
        res = await tool.execute(kind="xlsx", title="T", sheets=[], context=CTX)
        assert res["ok"] is False
        assert "sheets" in res["error"]

    @pytest.mark.asyncio
    async def test_empty_sheet_rejected(self, tool):
        res = await tool.execute(
            kind="xlsx", title="T", sheets=[{"name": "Empty"}], context=CTX)
        assert res["ok"] is False
        assert "sheets[0]" in res["error"]

    @pytest.mark.asyncio
    async def test_sheet_str_item_coerced(self, tool):
        res = await tool.execute(
            kind="xlsx", title="T",
            sheets=['{"name": "S", "headers": ["a"], "rows": [["1"]]}'],
            context=CTX,
        )
        assert res["ok"] is True

    @pytest.mark.asyncio
    async def test_sheet_rows_scalar_wrapped(self, tool):
        res = await tool.execute(
            kind="xlsx", title="T",
            sheets=[{"headers": ["a", "b"], "rows": [["x", 1], "scalar"]}],
            context=CTX,
        )
        assert res["ok"] is True

    @pytest.mark.asyncio
    async def test_sheet_caps(self, tool, monkeypatch):
        import agent.tools.builtin_tools.document_generator as dg
        monkeypatch.setattr(dg, "_MAX_SHEETS", 1)
        res = await tool.execute(
            kind="xlsx", title="T",
            sheets=[{"headers": ["a"], "rows": [[1]]},
                    {"headers": ["b"], "rows": [[2]]}],
            context=CTX,
        )
        assert res["ok"] is False
        assert "too many sheets" in res["error"]

    @pytest.mark.asyncio
    async def test_sheet_row_cap(self, tool, monkeypatch):
        import agent.tools.builtin_tools.document_generator as dg
        monkeypatch.setattr(dg, "_MAX_ROWS", 2)
        res = await tool.execute(
            kind="xlsx", title="T",
            sheets=[{"headers": ["a"], "rows": [[1], [2], [3]]}],
            context=CTX,
        )
        assert res["ok"] is False
        assert "too many rows" in res["error"]

    @pytest.mark.asyncio
    async def test_sheet_chart_needs_numeric_series(self, tool):
        res = await tool.execute(
            kind="xlsx", title="T",
            sheets=[{"headers": ["a"], "rows": [[1]],
                     "chart": {"type": "pie", "series": [{"values": ["x"]}]}}],
            context=CTX,
        )
        assert res["ok"] is False
        assert "numbers" in res["error"]


class TestIncrementalXlsx:
    @pytest.mark.asyncio
    async def test_create_add_edit_finalize_xlsx(self, tool):
        res = await tool.execute(kind="xlsx", title="Book", action="create", context=CTX)
        assert res["ok"] is True
        draft_id = res["content"].split("draft_id='")[1].split("'")[0]

        res = await tool.execute(action="add_sheets", draft_id=draft_id,
                                 sheets=[{"name": "S1", "headers": ["a"], "rows": [[1]]}],
                                 context=CTX)
        assert res["ok"] is True and "1/20" in res["content"]

        res = await tool.execute(
            action="edit", draft_id=draft_id,
            edits=[{"operation": "replace", "index": 0,
                    "sheet": {"name": "S1new", "headers": ["b"], "rows": [[2]]}}],
            context=CTX,
        )
        assert res["ok"] is True and "now has 1 sheets" in res["content"]

        res = await tool.execute(action="finalize", draft_id=draft_id, context=CTX)
        assert res["ok"] is True
        path = tool._store.resolve("conv-1", res["metadata"]["files"][0]["name"])
        from openpyxl import load_workbook
        wb = load_workbook(str(path))
        assert "S1new" in wb.sheetnames
        assert wb["S1new"]["A1"].value == "b"
        wb.close()

    @pytest.mark.asyncio
    async def test_add_sections_on_xlsx_draft_rejected(self, tool):
        res = await tool.execute(kind="xlsx", title="B", action="create", context=CTX)
        draft_id = res["content"].split("draft_id='")[1].split("'")[0]
        res = await tool.execute(action="add_sections", draft_id=draft_id,
                                 sections=[{"paragraphs": ["x"]}], context=CTX)
        assert res["ok"] is False
        assert "add_sheets" in res["error"]

    @pytest.mark.asyncio
    async def test_incremental_csv(self, tool):
        res = await tool.execute(kind="csv", title="C", action="create", context=CTX)
        draft_id = res["content"].split("draft_id='")[1].split("'")[0]
        res = await tool.execute(action="add_sheets", draft_id=draft_id,
                                 sheets=[{"headers": ["k"], "rows": [["v"]]}],
                                 context=CTX)
        assert res["ok"] is True
        res = await tool.execute(action="finalize", draft_id=draft_id, context=CTX)
        assert res["ok"] is True
        assert res["metadata"]["files"][0]["name"].endswith(".csv")


def _ws_ctx(ws_dir):
    return {
        "conversation_id": "conv-1",
        "assistant_name": "AssistantX",
        "agent_workspace_dir": str(ws_dir),
        "agent_workspace_cfg": {},
    }


class TestWorkspaceCopy:
    @pytest.mark.asyncio
    async def test_workspace_copy(self, tool, tmp_path):
        ws = tmp_path / "ws"
        res = await tool.execute(
            kind="docx", title="R",
            sections=[{"paragraphs": ["hello"]}],
            workspace_path="reports/q3.docx",
            context=_ws_ctx(ws))
        assert res["ok"] is True
        assert (ws / "reports" / "q3.docx").is_file()
        assert res["metadata"]["workspace_path"] == "reports/q3.docx"
        assert res["metadata"]["files"][0]["name"].endswith(".docx")
        assert "workspace" in res["content"].lower()

    @pytest.mark.asyncio
    async def test_workspace_extension_forced(self, tool, tmp_path):
        ws = tmp_path / "ws"
        res = await tool.execute(
            kind="docx", title="R", sections=[{"paragraphs": ["x"]}],
            workspace_path="r/x.txt", context=_ws_ctx(ws))
        assert res["ok"] is True
        assert (ws / "r" / "x.docx").is_file()
        assert not (ws / "r" / "x.txt").exists()

    @pytest.mark.asyncio
    async def test_workspace_disabled(self, tool, tmp_path):
        res = await tool.execute(
            kind="docx", title="R", sections=[{"paragraphs": ["x"]}],
            workspace_path="x.docx", context=CTX)
        assert res["ok"] is False
        assert "not enabled" in res["error"]
        conv_dir = tmp_path / "artifacts" / "conv-1"
        assert not conv_dir.exists() or not list(conv_dir.glob("*.docx"))

    @pytest.mark.asyncio
    async def test_workspace_traversal(self, tool, tmp_path):
        ws = tmp_path / "ws"
        res = await tool.execute(
            kind="docx", title="R", sections=[{"paragraphs": ["x"]}],
            workspace_path="../evil.docx", context=_ws_ctx(ws))
        assert res["ok"] is False
        assert "invalid workspace_path" in res["error"]
        assert not (tmp_path / "evil.docx").exists()

    @pytest.mark.asyncio
    async def test_workspace_quota_failure(self, tool, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "existing.txt").write_text("x")
        ctx = _ws_ctx(ws)
        ctx["agent_workspace_cfg"] = {"max_files": 1}
        res = await tool.execute(
            kind="docx", title="R", sections=[{"paragraphs": ["x"]}],
            workspace_path="new.docx", context=ctx)
        assert res["ok"] is True
        assert "Workspace copy failed" in res["content"]
        assert res["metadata"]["files"][0]["name"].endswith(".docx")
        assert "workspace_path" not in res["metadata"]
        assert not (ws / "new.docx").exists()

    @pytest.mark.asyncio
    async def test_draft_workspace_persist(self, tool, tmp_path):
        ctx = _ws_ctx(tmp_path / "ws")
        res = await tool.execute(kind="docx", title="D", action="create",
                                 workspace_path="out/d.docx", context=ctx)
        draft_id = res["content"].split("draft_id='")[1].split("'")[0]
        await tool.execute(action="add_sections", draft_id=draft_id,
                           sections=[{"paragraphs": ["body"]}], context=ctx)
        res = await tool.execute(action="finalize", draft_id=draft_id, context=ctx)
        assert res["ok"] is True
        assert (tmp_path / "ws" / "out" / "d.docx").is_file()

    @pytest.mark.asyncio
    async def test_draft_finalize_override(self, tool, tmp_path):
        ws = tmp_path / "ws"
        ctx = _ws_ctx(ws)
        res = await tool.execute(kind="docx", title="D", action="create",
                                 workspace_path="a.docx", context=ctx)
        draft_id = res["content"].split("draft_id='")[1].split("'")[0]
        await tool.execute(action="add_sections", draft_id=draft_id,
                           sections=[{"paragraphs": ["b"]}], context=ctx)
        res = await tool.execute(action="finalize", draft_id=draft_id,
                                 workspace_path="b.docx", context=ctx)
        assert res["ok"] is True
        assert (ws / "b.docx").is_file()
        assert not (ws / "a.docx").exists()

    @pytest.mark.asyncio
    async def test_draft_finalize_clears_stored_path(self, tool, tmp_path):
        ws = tmp_path / "ws"
        ctx = _ws_ctx(ws)
        res = await tool.execute(kind="docx", title="D", action="create",
                                 workspace_path="a.docx", context=ctx)
        draft_id = res["content"].split("draft_id='")[1].split("'")[0]
        await tool.execute(action="add_sections", draft_id=draft_id,
                           sections=[{"paragraphs": ["b"]}], context=ctx)
        res = await tool.execute(action="finalize", draft_id=draft_id,
                                 workspace_path="", context=ctx)
        assert res["ok"] is True
        assert not (ws / "a.docx").exists()

    def test_conflict_key_workspace_path(self, tool):
        assert tool.call_conflict_key({"workspace_path": "r/x.docx"}) == "ws:r/x.docx"
        assert tool.call_conflict_key(
            {"draft_id": "doc-1", "workspace_path": "x.docx"}) == "draft:doc-1"
        assert tool.call_conflict_key({}) is None


class TestRobustness:
    """Guards against spec leaks / malformed shapes producing garbage docs."""

    @pytest.mark.asyncio
    async def test_spec_json_in_paragraphs_rejected(self, tool):
        res = await tool.execute(
            kind="docx", title="T",
            sections=[{"paragraphs": ['[{"heading": "x", "level": 1, "paragraphs": ["p"]}]']}],
            context=CTX,
        )
        assert res["ok"] is False
        assert "embedded" in res["error"]

    @pytest.mark.asyncio
    async def test_plain_json_text_allowed(self, tool):
        res = await tool.execute(
            kind="docx", title="T",
            sections=[{"paragraphs": ['{"id": "mail_01", "sender": "a@b.de"}']}],
            context=CTX,
        )
        assert res["ok"] is True
        file_meta = res["metadata"]["files"][0]
        path = tool._store.resolve("conv-1", file_meta["name"])
        from docx import Document
        doc = Document(str(path))
        assert any("mail_01" in p.text for p in doc.paragraphs)

    @pytest.mark.asyncio
    async def test_sections_as_json_string_unwrapped(self, tool):
        res = await tool.execute(
            kind="docx", title="T",
            sections='[{"heading": "H", "paragraphs": ["p1"]}]',
            context=CTX,
        )
        assert res["ok"] is True
        file_meta = res["metadata"]["files"][0]
        path = tool._store.resolve("conv-1", file_meta["name"])
        from docx import Document
        doc = Document(str(path))
        assert any(p.text == "H" for p in doc.paragraphs)

    @pytest.mark.asyncio
    async def test_sections_scalar_rejected(self, tool):
        res = await tool.execute(kind="docx", title="T", sections=42, context=CTX)
        assert res["ok"] is False
        assert "must be a list" in res["error"]

    @pytest.mark.asyncio
    async def test_empty_section_rejected(self, tool):
        res = await tool.execute(
            kind="docx", title="T",
            sections=[{"paragraphs": ["ok"]}, {}],
            context=CTX,
        )
        assert res["ok"] is False
        assert "no content" in res["error"]

    @pytest.mark.asyncio
    async def test_markdown_runs_rendered(self, tool):
        res = await tool.execute(
            kind="docx", title="T",
            sections=[{"paragraphs": ["plain **fett** und *kursiv* mit `code`"]}],
            context=CTX,
        )
        assert res["ok"] is True
        file_meta = res["metadata"]["files"][0]
        path = tool._store.resolve("conv-1", file_meta["name"])
        from docx import Document
        doc = Document(str(path))
        para = next(p for p in doc.paragraphs if "fett" in p.text)
        assert "**" not in para.text
        bold = [r.text for r in para.runs if r.bold]
        assert bold == ["fett"]

    @pytest.mark.asyncio
    async def test_bullet_dict_text_rendered(self, tool):
        res = await tool.execute(
            kind="docx", title="T",
            sections=[{"bullets": ["eins", "zwei"]}],
            context=CTX,
        )
        assert res["ok"] is True
        file_meta = res["metadata"]["files"][0]
        path = tool._store.resolve("conv-1", file_meta["name"])
        from docx import Document
        doc = Document(str(path))
        texts = [p.text for p in doc.paragraphs]
        assert "eins" in texts and "zwei" in texts
        assert not any("{'text'" in t or '"text"' in t for t in texts)

    @pytest.mark.asyncio
    async def test_first_section_title_deduped(self, tool):
        res = await tool.execute(
            kind="docx", title="Mein Bericht",
            sections=[{"heading": "Mein Bericht", "paragraphs": ["p"]},
                      {"heading": "Zweiter Teil", "paragraphs": ["q"]}],
            context=CTX,
        )
        assert res["ok"] is True
        file_meta = res["metadata"]["files"][0]
        path = tool._store.resolve("conv-1", file_meta["name"])
        from docx import Document
        doc = Document(str(path))
        # "Mein Bericht" appears once (title style) — the duplicated
        # first-section heading is skipped, "Zweiter Teil" rendered as heading.
        texts = [p.text for p in doc.paragraphs]
        assert texts.count("Mein Bericht") == 1
        assert any(p.style.name.startswith("Heading") and p.text == "Zweiter Teil"
                   for p in doc.paragraphs)

    @pytest.mark.asyncio
    async def test_heading_markdown_in_paragraph(self, tool):
        res = await tool.execute(
            kind="docx", title="T",
            sections=[{"paragraphs": ["## Unterkapitel", "text"]}],
            context=CTX,
        )
        assert res["ok"] is True
        file_meta = res["metadata"]["files"][0]
        path = tool._store.resolve("conv-1", file_meta["name"])
        from docx import Document
        doc = Document(str(path))
        assert any(p.style.name == "Heading 2" and p.text == "Unterkapitel" for p in doc.paragraphs)


class TestDeliver:
    @pytest.mark.asyncio
    async def test_deliver_false_omits_files_metadata(self, tool):
        res = await tool.execute(
            kind="docx", title="Internal",
            sections=[{"heading": "S", "paragraphs": ["p"]}],
            deliver=False, context=CTX,
        )
        assert res["ok"] is True
        assert "files" not in res["metadata"]
        assert "not shown in the chat" in res["content"]
        # the artifact is still stored so publish can attach it later
        entry = tool._store.get_entry("conv-1", "internal.docx")
        assert entry is not None and entry["kind"] == "docx"

    @pytest.mark.asyncio
    async def test_deliver_default_true_attaches(self, tool):
        res = await tool.execute(
            kind="docx", title="Shown",
            sections=[{"heading": "S", "paragraphs": ["p"]}],
            context=CTX,
        )
        assert res["ok"] is True
        assert res["metadata"]["files"][0]["name"] == "shown.docx"

    @pytest.mark.asyncio
    async def test_publish_emits_files_for_existing_artifact(self, tool):
        await tool.execute(
            kind="docx", title="Later", filename="later.docx",
            sections=[{"heading": "S", "paragraphs": ["p"]}],
            deliver=False, context=CTX,
        )
        res = await tool.execute(action="publish", filename="later.docx", context=CTX)
        assert res["ok"] is True
        files = res["metadata"]["files"]
        assert files[0]["name"] == "later.docx"
        assert files[0]["conversation_id"] == "conv-1"

    @pytest.mark.asyncio
    async def test_publish_unknown_filename_errors(self, tool):
        res = await tool.execute(action="publish", filename="nope.docx", context=CTX)
        assert res["ok"] is False
        assert "no artifact" in res["content"]

    @pytest.mark.asyncio
    async def test_publish_requires_filename(self, tool):
        res = await tool.execute(action="publish", context=CTX)
        assert res["ok"] is False
        assert "filename" in res["content"]

    @pytest.mark.asyncio
    async def test_draft_deliver_stored_and_finalize_override(self, tool):
        res = await tool.execute(
            action="create", kind="docx", title="Draft",
            deliver=False, context=CTX,
        )
        draft_id = res["content"].split("draft_id='")[1].split("'")[0]
        await tool.execute(
            action="add_sections", draft_id=draft_id,
            sections=[{"heading": "S", "paragraphs": ["p"]}],
            context=CTX,
        )
        res = await tool.execute(action="finalize", draft_id=draft_id, context=CTX)
        assert res["ok"] is True
        assert "files" not in res["metadata"]  # stored deliver=false wins

        # finalize deliver=true overrides the stored flag
        res = await tool.execute(
            action="create", kind="docx", title="Draft2",
            deliver=False, context=CTX,
        )
        draft_id = res["content"].split("draft_id='")[1].split("'")[0]
        await tool.execute(
            action="add_sections", draft_id=draft_id,
            sections=[{"heading": "S", "paragraphs": ["p"]}],
            context=CTX,
        )
        res = await tool.execute(
            action="finalize", draft_id=draft_id, deliver=True, context=CTX)
        assert res["ok"] is True
        assert res["metadata"]["files"][0]["name"] == "draft2.docx"

    @pytest.mark.asyncio
    async def test_same_filename_updates_artifact_in_place(self, tool):
        kwargs = dict(kind="docx", title="V", filename="v.docx", context=CTX)
        r1 = await tool.execute(sections=[{"heading": "S", "paragraphs": ["first"]}], **kwargs)
        r2 = await tool.execute(sections=[{"heading": "S", "paragraphs": ["second version"]}], **kwargs)
        assert r1["metadata"]["files"][0]["name"] == "v.docx"
        assert r2["metadata"]["files"][0]["name"] == "v.docx"  # no v-2.docx
        names = [k for k in tool._store.load_index("conv-1") if k.startswith("v")]
        assert names == ["v.docx"]
        assert "updated_at" in r2["metadata"]["files"][0]
