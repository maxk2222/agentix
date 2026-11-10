"""Document generator builtin tool.

Creates .docx/.pptx/.pdf/.xlsx/.csv files from a structured JSON spec with a
small design layer (accent colours, fonts, layout variants), native editable
PowerPoint/Excel charts, matplotlib-rendered charts (docx + ``render:
"image"``) and shape-based flow diagrams. Files are stored via
``ArtifactStore`` and surfaced to the chat UI through ``metadata.files``.
"""

from __future__ import annotations

import asyncio
import datetime
import io
import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.agent_workspace import AgentWorkspaceStore
from app.artifacts import ArtifactStore, sanitize_filename

logger = logging.getLogger(__name__)

_MAX_SECTIONS = 100
_MAX_SLIDES = 60
_MAX_CHARTS = 20
_MAX_TEXT_CHARS = 200_000
_MAX_NODES = 30
_MAX_SHEETS = 20
_MAX_ROWS = 2000
_MAX_COLUMNS = 60
_MAX_TOTAL_CELLS = 150_000

_DEFAULT_THEME: Dict[str, str] = {
    "accent": "0A66C2",
    "accent_dark": "084B96",
    "accent_light": "D6E8F9",
    "heading_font": "Calibri Light",
    "body_font": "Calibri",
}


def _slug(text: str) -> str:
    slug = re.sub(r"[^\w\-]+", "-", (text or "").lower()).strip("-")
    return slug[:60] or "document"


def _hex(value: str, default: str) -> str:
    v = re.sub(r"[^0-9a-fA-F]", "", value or "")
    return v[:6].upper() if len(v) >= 6 else default


def _palette(accent: str) -> List[str]:
    """Accent-first colour ramp for chart series/points."""
    try:
        r, g, b = int(accent[0:2], 16), int(accent[2:4], 16), int(accent[4:6], 16)
    except Exception:
        r, g, b = 10, 102, 194

    def shade(f: float) -> str:
        nr = max(0, min(255, int(r + (255 - r) * f if f > 0 else r * (1 + f))))
        ng = max(0, min(255, int(g + (255 - g) * f if f > 0 else g * (1 + f))))
        nb = max(0, min(255, int(b + (255 - b) * f if f > 0 else b * (1 + f))))
        return f"{nr:02X}{ng:02X}{nb:02X}"

    return [accent, shade(0.35), shade(-0.25), shade(0.6), shade(-0.45), "8A8F98", "C7CDD4"]


def _theme(context: Optional[Dict[str, Any]]) -> Dict[str, str]:
    cfg = ((context or {}).get("tool_config") or {}).get("theme") or {}
    theme = dict(_DEFAULT_THEME)
    for key in ("accent", "accent_dark", "accent_light"):
        if cfg.get(key):
            theme[key] = _hex(str(cfg[key]), theme[key])
    for key in ("heading_font", "body_font"):
        if cfg.get(key):
            theme[key] = str(cfg[key])[:60]
    return theme


def _err(msg: str) -> Dict[str, Any]:
    return {"ok": False, "error": msg, "content": msg}


_DRAFT_TTL_SECONDS = 24 * 3600
_INCREMENTAL_HINT = " For large documents build incrementally: action='create' -> 'add_sections'/'add_slides'/'add_sheets' -> 'finalize'."


def _json_or_str(value: Any) -> Any:
    """Parse a JSON-encoded string, otherwise return the value unchanged."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                return json.loads(stripped)
            except (ValueError, TypeError):
                return value
    return value


def _spec_kind(kind: str) -> str:
    """Return the underlying spec type: 'pdf' renders the docx sections spec,
    'csv' renders the first sheet of the xlsx sheets spec."""
    return {"pdf": "docx", "csv": "xlsx"}.get(kind, kind)


_VALID_KINDS = ("docx", "pptx", "pdf", "xlsx", "csv")
_CONTAINER_FOR_KIND = {"docx": "sections", "pptx": "slides", "xlsx": "sheets"}
_PAYLOAD_KEY_FOR_KIND = {"docx": "section", "pptx": "slide", "xlsx": "sheet"}


def _normalize_chart(chart: Any, loc: str) -> Optional[str]:
    """Coerce a chart spec's series/values/categories in place.

    Returns an error string for unrepairable shapes, None on success."""
    if not isinstance(chart, dict):
        return None
    series = _json_or_str(chart.get("series"))
    if isinstance(series, dict):
        series = [series]
    if series is not None and not isinstance(series, list):
        return f"{loc}.chart.series must be a list"
    if isinstance(series, list):
        fixed = []
        for j, s in enumerate(series):
            if not isinstance(s, dict):
                return f"{loc}.chart.series[{j}] must be an object"
            vals = _json_or_str(s.get("values"))
            if vals is not None and not isinstance(vals, list):
                vals = [vals]
            s["values"] = vals
            fixed.append(s)
        series = fixed
    chart["series"] = series
    cats = _json_or_str(chart.get("categories"))
    if cats is not None and not isinstance(cats, list):
        cats = [cats]
    chart["categories"] = cats
    return None


_SPEC_FIELD_KEYS = {
    "heading", "paragraphs", "bullets", "slides", "sections", "sheets",
    "layout", "series", "diagram", "categories", "level", "notes",
    "column_widths", "number_formats", "freeze_header", "auto_filter",
}
_LEAK_MSG = (
    "contains embedded document-spec JSON — pass content via the structured "
    "'sections'/'slides'/'sheets' fields, not as JSON text"
)


def _looks_like_spec(value: Any) -> bool:
    """True when parsed JSON resembles an embedded document spec, not data.

    Data-shaped JSON (e.g. ``{"id": "mail_01"}``) stays allowed in text —
    only values carrying actual spec field names are flagged.
    """
    if isinstance(value, dict):
        keys = set(value)
        return len(keys & _SPEC_FIELD_KEYS) >= 2 or {"headers", "rows"} <= keys
    if isinstance(value, list) and value:
        return any(
            _looks_like_spec(v) for v in value if isinstance(v, (dict, list))
        )
    return False


def _spec_leak_error(spec: Dict[str, Any]) -> Optional[str]:
    """Scan text-bearing fields for spec JSON dumped as a string (or dict).

    Models occasionally wrap the whole spec inside e.g. a paragraphs element;
    without this check it would be rendered verbatim into the document.
    """
    def leaky(value: Any) -> bool:
        if isinstance(value, dict):
            return _looks_like_spec(value)
        return isinstance(value, str) and _looks_like_spec(_json_or_str(value))

    for container_name in ("sections", "slides"):
        for i, item in enumerate(spec.get(container_name) or []):
            loc = f"{container_name}[{i}]"
            for field in ("paragraphs", "bullets"):
                for j, el in enumerate(item.get(field) or []):
                    if isinstance(el, dict):
                        if _looks_like_spec(el) or leaky(el.get("text")):
                            return f"{loc}.{field}[{j}] {_LEAK_MSG}"
                    elif leaky(el):
                        return f"{loc}.{field}[{j}] {_LEAK_MSG}"
            for field in ("title", "body", "notes"):
                if leaky(item.get(field)):
                    return f"{loc}.{field} {_LEAK_MSG}"
            table = item.get("table") or {}
            cells = list(table.get("headers") or [])
            for r in table.get("rows") or []:
                cells.extend(r if isinstance(r, list) else [r])
            for j, el in enumerate(cells):
                if leaky(el):
                    return f"{loc}.table cell {j} {_LEAK_MSG}"
    for i, item in enumerate(spec.get("sheets") or []):
        cells = list(item.get("headers") or [])
        for r in item.get("rows") or []:
            cells.extend(r if isinstance(r, list) else [r])
        for j, el in enumerate(cells):
            if leaky(el):
                return f"sheets[{i}] cell {j} {_LEAK_MSG}"
    return None


_MD_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
_INLINE_MD = re.compile(r"(\*\*[^*]+?\*\*|\*[^*\n]+?\*|`[^`\n]+?`)")


def _add_md_runs(p, text: Any, font: str) -> None:
    """Write ``text`` into paragraph ``p``, honoring **bold**, *italic*, `code`."""
    for chunk in _INLINE_MD.split(str(text)):
        if not chunk:
            continue
        out, bold, italic, mono = chunk, False, False, False
        if len(chunk) > 4 and chunk.startswith("**") and chunk.endswith("**"):
            out, bold = chunk[2:-2], True
        elif len(chunk) > 2 and chunk.startswith("*") and chunk.endswith("*"):
            out, italic = chunk[1:-1], True
        elif len(chunk) > 2 and chunk.startswith("`") and chunk.endswith("`"):
            out, mono = chunk[1:-1], True
        run = p.add_run(out)
        run.font.name = "Consolas" if mono else font
        run.font.bold = bold or None
        run.font.italic = italic or None


def _normalize_items(spec: Dict[str, Any], kind: str) -> Optional[str]:
    """Repair common model arg-shape issues inside sections/slides/sheets specs.

    Models often serialize nested structures as JSON strings or pass bare
    strings/ints where objects are declared. Coerce what we can and return a
    precise indexed error for the rest so the model can self-correct instead
    of hitting an AttributeError.
    """
    for container_name in ("sections", "slides", "sheets"):
        items = spec.get(container_name)
        if isinstance(items, str):
            items = _json_or_str(items)
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            if items is not None:
                return f"'{container_name}' must be a list (got {type(items).__name__})"
            continue
        spec[container_name] = items
        normalized: List[Dict[str, Any]] = []
        for i, item in enumerate(items):
            if isinstance(item, str):
                item = _json_or_str(item)
                if not isinstance(item, dict):
                    if container_name == "sections":
                        item = {"paragraphs": [str(item)]}
                    elif container_name == "slides":
                        item = {"bullets": [str(item)]}
                    elif isinstance(item, list):
                        item = {"rows": item}
                    else:
                        item = {"name": str(item)}
            elif container_name == "sheets" and isinstance(item, list):
                item = {"rows": item}
            if not isinstance(item, dict):
                return f"{container_name}[{i}] must be an object (got {type(item).__name__})"
            normalized.append(item)
        spec[container_name] = normalized

    for container_name in ("sections", "slides"):
        for i, item in enumerate(spec.get(container_name) or []):
            loc = f"{container_name}[{i}]"
            for key in ("paragraphs", "bullets", "categories"):
                val = _json_or_str(item.get(key))
                if val is not None and not isinstance(val, list):
                    val = [val]
                if key == "bullets" and isinstance(val, list):
                    val = [b if isinstance(b, dict) else {"text": str(b)} for b in val]
                item[key] = val
            for key in ("table", "chart", "diagram"):
                val = _json_or_str(item.get(key))
                if val is not None and not isinstance(val, dict):
                    return f"{loc}.{key} must be an object"
                item[key] = val

            table = item.get("table")
            if isinstance(table, dict):
                headers = _json_or_str(table.get("headers"))
                if headers is not None and not isinstance(headers, list):
                    headers = [headers]
                table["headers"] = headers
                rows = _json_or_str(table.get("rows"))
                if rows is not None and not isinstance(rows, list):
                    return f"{loc}.table.rows must be a list of rows"
                if isinstance(rows, list):
                    table["rows"] = [r if isinstance(r, list) else [r] for r in rows]
                else:
                    table["rows"] = rows

            chart_error = _normalize_chart(item.get("chart"), loc)
            if chart_error:
                return chart_error

            diagram = item.get("diagram")
            if isinstance(diagram, dict):
                nodes = _json_or_str(diagram.get("nodes"))
                if nodes is not None and not isinstance(nodes, list):
                    return f"{loc}.diagram.nodes must be a list"
                if isinstance(nodes, list):
                    fixed_nodes = []
                    for n in nodes:
                        if isinstance(n, dict):
                            if n.get("id") is None and n.get("label") is not None:
                                n = {**n, "id": n["label"]}
                            fixed_nodes.append(n)
                        else:
                            fixed_nodes.append({"id": str(n), "label": str(n)})
                    nodes = fixed_nodes
                diagram["nodes"] = nodes
                edges = _json_or_str(diagram.get("edges"))
                if edges is not None and not isinstance(edges, list):
                    return f"{loc}.diagram.edges must be a list"
                if isinstance(edges, list):
                    fixed_edges = []
                    for e in edges:
                        if isinstance(e, dict):
                            src = e.get("from", e.get("source"))
                            dst = e.get("to", e.get("target"))
                            if src is not None and dst is not None:
                                fixed_edges.append([src, dst])
                            else:
                                return f"{loc}.diagram.edges entry must be a [from, to] pair"
                        elif isinstance(e, (list, tuple)) and len(e) >= 2:
                            fixed_edges.append([e[0], e[1]])
                        else:
                            return f"{loc}.diagram.edges entry must be a [from, to] pair"
                    edges = fixed_edges
                diagram["edges"] = edges

    for i, item in enumerate(spec.get("sheets") or []):
        loc = f"sheets[{i}]"
        headers = _json_or_str(item.get("headers"))
        if headers is not None and not isinstance(headers, list):
            headers = [headers]
        item["headers"] = headers
        rows = _json_or_str(item.get("rows"))
        if rows is not None and not isinstance(rows, list):
            return f"{loc}.rows must be a list of rows"
        if isinstance(rows, list):
            item["rows"] = [r if isinstance(r, list) else [r] for r in rows]
        else:
            item["rows"] = rows
        for key in ("column_widths", "number_formats"):
            val = _json_or_str(item.get(key))
            if val is not None and not isinstance(val, (list, dict)):
                return f"{loc}.{key} must be a list or object"
            item[key] = val
        for key in ("freeze_header", "auto_filter"):
            val = item.get(key)
            if isinstance(val, str):
                item[key] = val.strip().lower() not in ("false", "0", "no", "off")
        chart = _json_or_str(item.get("chart"))
        if chart is not None and not isinstance(chart, dict):
            return f"{loc}.chart must be an object"
        item["chart"] = chart
        chart_error = _normalize_chart(chart, loc)
        if chart_error:
            return chart_error
    return _spec_leak_error(spec)


# --------------------------------------------------------------------------
# matplotlib image charts (docx + pptx render:"image")
# --------------------------------------------------------------------------

def _render_chart_png(chart: Dict[str, Any], theme: Dict[str, str]) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chart_type = str(chart.get("type", "column")).lower()
    categories = [str(c) for c in (chart.get("categories") or [])][:20]
    series = chart.get("series") or []
    title = str(chart.get("title") or "").strip()
    colors = ["#" + c for c in _palette(theme["accent"])]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    fig.patch.set_alpha(0)
    x = list(range(len(categories)))

    if chart_type in ("pie", "doughnut"):
        vals = [float(v) for v in (series[0].get("values") if series else [])][:20]
        wedges, _, autotexts = ax.pie(
            vals, labels=categories or None, autopct="%1.0f%%",
            colors=colors, pctdistance=0.8,
            wedgeprops={"width": 0.42} if chart_type == "doughnut" else {},
            textprops={"fontsize": 9},
        )
        for t in autotexts:
            t.set_fontsize(8)
            t.set_color("white")
        ax.set_aspect("equal")
    else:
        stacked = chart_type.endswith("_stacked")
        base_type = chart_type.replace("_stacked", "")
        bottoms = [0.0] * len(categories)
        for i, s in enumerate(series[:8]):
            vals = [float(v) for v in (s.get("values") or [])][: len(categories)]
            vals += [0.0] * (len(categories) - len(vals))
            label = str(s.get("name") or f"Series {i + 1}")
            color = colors[i % len(colors)]
            if base_type == "bar":
                fn = ax.barh
                if stacked:
                    fn(categories, vals, left=bottoms, label=label, color=color)
                else:
                    w = 0.8 / max(len(series), 1)
                    fn([p + i * w - 0.4 + w / 2 for p in x], vals, w,
                       label=label, color=color)
                    ax.set_yticks(x)
                    ax.set_yticklabels(categories, fontsize=8)
            elif base_type in ("line", "line_markers"):
                ax.plot(categories, vals, marker="o" if base_type == "line_markers" else None,
                        label=label, color=color, linewidth=2)
            elif base_type == "area":
                ax.fill_between(x, vals, alpha=0.5, label=label, color=color)
                ax.set_xticks(x)
                ax.set_xticklabels(categories, fontsize=8)
            else:  # column
                w = 0.8 / max(len(series), 1)
                if stacked:
                    ax.bar(x, vals, w, bottom=bottoms, label=label, color=color)
                else:
                    ax.bar([p + i * w - 0.4 + w / 2 for p in x], vals, w, label=label, color=color)
                ax.set_xticks(x)
                ax.set_xticklabels(categories, fontsize=8)
            if stacked:
                bottoms = [b + v for b, v in zip(bottoms, vals)]
        if len(series) > 1:
            ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        if base_type == "bar" and categories:
            pass  # barh already uses category labels
        elif base_type not in ("column", "area") and categories and base_type.startswith("line"):
            ax.set_xticks(x)
            ax.set_xticklabels(categories, fontsize=8)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    return buf.getvalue()


def _png_via_graphviz(diagram: Dict[str, Any]) -> Optional[bytes]:
    """Render the diagram through graphviz ``dot`` when available."""
    dot_bin = shutil.which("dot")
    if dot_bin is None:
        return None
    nodes = diagram.get("nodes") or []
    edges = diagram.get("edges") or []
    direction = "LR" if str(diagram.get("direction", "h")).lower() != "v" else "TB"
    lines = [f'digraph G {{ rankdir={direction}; node [shape=box, style="rounded,filled", fillcolor="#D6E8F9", fontname="Helvetica"];']
    for node in nodes:
        label = str(node.get("label", node.get("id", ""))).replace('"', "'")
        lines.append(f'  "{node.get("id")}" [label="{label}"];')
    for edge in edges:
        if len(edge) >= 2:
            lines.append(f'  "{edge[0]}" -> "{edge[1]}";')
    lines.append("}")
    try:
        result = subprocess.run(
            [dot_bin, "-Tpng"], input="\n".join(lines).encode(),
            capture_output=True, timeout=20,
        )
        return result.stdout if result.returncode == 0 and result.stdout else None
    except Exception:
        return None


# --------------------------------------------------------------------------
# docx
# --------------------------------------------------------------------------

def _build_docx(title: str, spec: Dict[str, Any], theme: Dict[str, str]) -> bytes:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt, RGBColor

    accent = RGBColor.from_string(theme["accent"])
    doc = Document()
    props = doc.core_properties
    props.title = title
    if spec.get("author"):
        props.author = str(spec["author"])[:120]
    if spec.get("subject"):
        props.subject = str(spec["subject"])[:200]

    for level in (1, 2, 3):
        try:
            style = doc.styles[f"Heading {level}"]
            style.font.name = theme["heading_font"]
            style.font.color.rgb = accent
        except Exception:
            pass
    try:
        normal = doc.styles["Normal"]
        normal.font.name = theme["body_font"]
        normal.font.size = Pt(11)
    except Exception:
        pass

    title_para = doc.add_heading(title, level=0)
    for run in title_para.runs:
        run.font.color.rgb = accent
    if spec.get("subtitle"):
        sub = doc.add_paragraph(str(spec["subtitle"]))
        sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph("")

    norm_title = re.sub(r"\s+", " ", str(title)).strip().lower()
    for si, section in enumerate(spec.get("sections") or []):
        heading = str(section.get("heading") or "").strip()
        if (si == 0 and heading
                and re.sub(r"\s+", " ", heading).lower() == norm_title):
            heading = ""  # first section repeats the title-page heading
        if heading:
            doc.add_heading(heading, level=min(3, max(1, int(section.get("level") or 1))))
        for para in (section.get("paragraphs") or []):
            m = _MD_HEADING.match(str(para).strip())
            if m:
                doc.add_heading(m.group(2), level=len(m.group(1)))
                continue
            p = doc.add_paragraph()
            _add_md_runs(p, para, theme["body_font"])
        for bullet in (section.get("bullets") or []):
            text = bullet.get("text", "") if isinstance(bullet, dict) else bullet
            p = doc.add_paragraph(style="List Bullet")
            _add_md_runs(p, text, theme["body_font"])
        table = section.get("table") or {}
        headers, rows = table.get("headers") or [], table.get("rows") or []
        if headers and rows:
            grid = doc.add_table(rows=len(rows) + 1, cols=len(headers))
            try:
                grid.style = "Table Grid"
            except Exception:
                pass
            for j, h in enumerate(headers):
                cell = grid.rows[0].cells[j]
                cell.text = ""
                _add_md_runs(cell.paragraphs[0], h, theme["body_font"])
                for p in cell.paragraphs:
                    for run in p.runs:
                        run.font.bold = True
                        run.font.color.rgb = RGBColor.from_string("FFFFFF")
                shading = cell._tc.get_or_add_tcPr().makeelement(
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}shd",
                    {"{http://schemas.openxmlformats.org/wordprocessingml/2006/main}fill": theme["accent"]},
                )
                cell._tc.get_or_add_tcPr().append(shading)
            for i, row in enumerate(rows):
                for j, val in enumerate(list(row)[: len(headers)]):
                    cell = grid.rows[i + 1].cells[j]
                    cell.text = ""
                    _add_md_runs(cell.paragraphs[0], val, theme["body_font"])
        chart = section.get("chart") or {}
        if chart:
            png = _render_chart_png(chart, theme)
            doc.add_picture(io.BytesIO(png), width=Inches(6.0))

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# --------------------------------------------------------------------------
# pptx design layer
# --------------------------------------------------------------------------

def _tb(slide, left, top, width, height):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    return tf


def _set_para(p, text: str, *, size: int, bold: bool, color: str, font: str, level: int = 0, bullet: bool = False):
    from pptx.util import Pt
    from pptx.dml.color import RGBColor

    p.level = max(0, min(level, 4))
    run = p.add_run()
    run.text = ("• " if bullet and level == 0 else ("    " * level + "– " if bullet else "")) + text
    f = run.font
    f.size = Pt(size)
    f.bold = bold
    f.name = font
    f.color.rgb = RGBColor.from_string(color)


def _fill_bullets(tf, bullets: List[Any], theme: Dict[str, str], base_size: int = 18) -> None:
    first = True
    for item in bullets:
        if isinstance(item, dict):
            text, level = str(item.get("text", "")), int(item.get("level") or 0)
        else:
            text, level = str(item), 0
        if not text.strip():
            continue
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        _set_para(p, text, size=base_size if level == 0 else base_size - 2,
                  bold=False, color="1F2937", font=theme["body_font"], level=level, bullet=True)


def _accent_bar(slide, theme: Dict[str, str], slide_w) -> None:
    from pptx.util import Inches
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE

    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, slide_w, Inches(0.12))
    bar.fill.solid()
    bar.fill.fore_color.rgb = RGBColor.from_string(theme["accent"])
    bar.line.fill.background()


def _title_header(slide, title: str, theme: Dict[str, str], slide_w) -> None:
    from pptx.util import Inches, Pt

    _accent_bar(slide, theme, slide_w)
    tf = _tb(slide, Inches(0.6), Inches(0.3), slide_w - Inches(1.2), Inches(0.95))
    _set_para(tf.paragraphs[0], title, size=30, bold=True,
              color=theme["accent_dark"], font=theme["heading_font"])


def _add_native_chart(slide, chart: Dict[str, Any], theme: Dict[str, str], left, top, width, height) -> None:
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
    from pptx.dml.color import RGBColor

    type_map = {
        "bar": XL_CHART_TYPE.BAR_CLUSTERED,
        "bar_stacked": XL_CHART_TYPE.BAR_STACKED,
        "column": XL_CHART_TYPE.COLUMN_CLUSTERED,
        "column_stacked": XL_CHART_TYPE.COLUMN_STACKED,
        "line": XL_CHART_TYPE.LINE,
        "line_markers": XL_CHART_TYPE.LINE_MARKERS,
        "pie": XL_CHART_TYPE.PIE,
        "doughnut": XL_CHART_TYPE.DOUGHNUT,
        "area": XL_CHART_TYPE.AREA,
    }
    chart_type = str(chart.get("type", "column")).lower()
    xl_type = type_map.get(chart_type, XL_CHART_TYPE.COLUMN_CLUSTERED)

    data = CategoryChartData()
    data.categories = [str(c) for c in (chart.get("categories") or [])][:20]
    for s in (chart.get("series") or [])[:8]:
        data.add_series(str(s.get("name") or "Series"), tuple(float(v) for v in (s.get("values") or [])))

    frame = slide.shapes.add_chart(xl_type, left, top, width, height, data)
    ch = frame.chart
    try:
        ch.has_title = bool(chart.get("title"))
        if chart.get("title"):
            ch.chart_title.text_frame.text = str(chart["title"])
        if chart_type not in ("pie", "doughnut") and len(data.series) > 1:
            ch.has_legend = True
            ch.legend.position = XL_LEGEND_POSITION.BOTTOM
            ch.legend.include_in_layout = False
        colors = _palette(theme["accent"])
        if chart_type in ("pie", "doughnut"):
            pass  # per-point colours need part-level XML; theme default is acceptable
        else:
            for i, series in enumerate(ch.series):
                series.format.fill.solid()
                series.format.fill.fore_color.rgb = RGBColor.from_string(colors[i % len(colors)])
        plot = ch.plots[0]
        plot.has_data_labels = True
        fmt = str(chart.get("number_format") or "General")
        plot.data_labels.number_format = fmt
        plot.data_labels.number_format_is_linked = False
    except Exception as exc:
        logger.debug("Native chart styling skipped: %s", exc)


def _add_picture_chart(slide, chart: Dict[str, Any], theme: Dict[str, str], left, top, width) -> None:
    png = _render_chart_png(chart, theme)
    slide.shapes.add_picture(io.BytesIO(png), left, top, width=width)


def _arrow(connector) -> None:
    """Add a triangle tail-end arrowhead to a connector (raw XML — python-pptx has no API)."""
    from pptx.oxml.ns import qn

    ln = connector.line._get_or_add_ln()
    tail = ln.find(qn("a:tailEnd"))
    if tail is None:
        from lxml import etree
        tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    tail.set("w", "med")
    tail.set("len", "med")


def _build_diagram(slide, diagram: Dict[str, Any], theme: Dict[str, str], left, top, width, height) -> None:
    from pptx.util import Inches, Pt, Emu
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR

    png = _png_via_graphviz(diagram)
    if png is not None:
        slide.shapes.add_picture(io.BytesIO(png), left, top, height=height)
        return

    nodes = [n for n in (diagram.get("nodes") or []) if n.get("id") is not None][:_MAX_NODES]
    edges = [e for e in (diagram.get("edges") or []) if len(e) >= 2]
    if not nodes:
        return
    direction = str(diagram.get("direction", "h")).lower()

    # layered layout: layer of a node = longest path from a source
    id_to_node = {str(n["id"]): n for n in nodes}
    incoming: Dict[str, int] = {str(n["id"]): 0 for n in nodes}
    children: Dict[str, List[str]] = {}
    for edge in edges:
        src, dst = str(edge[0]), str(edge[1])
        if src in id_to_node and dst in id_to_node:
            children.setdefault(src, []).append(dst)
            incoming[dst] = incoming.get(dst, 0) + 1
    layers: Dict[str, int] = {}
    queue = [nid for nid, deg in incoming.items() if deg == 0] or list(id_to_node)[:1]
    seen = set()
    while queue:
        nid = queue.pop(0)
        if nid in seen:
            continue
        seen.add(nid)
        for child in children.get(nid, []):
            layers[child] = max(layers.get(child, 0), layers.get(nid, 0) + 1)
            incoming[child] -= 1
            if incoming[child] <= 0:
                queue.append(child)
    for nid in id_to_node:
        layers.setdefault(nid, 0)
    by_layer: Dict[int, List[str]] = {}
    for nid, layer in layers.items():
        by_layer.setdefault(layer, []).append(nid)

    max_layer_width = max((len(ids) for ids in by_layer.values()), default=1)
    box_w = Inches(2.0) if direction != "v" else min(Inches(2.4), width / max_layer_width - Inches(0.3))
    box_h = Inches(0.9)
    positions: Dict[str, tuple] = {}
    max_layer = max(by_layer, default=0)
    for layer, ids in by_layer.items():
        count = len(ids)
        for i, nid in enumerate(ids):
            if direction == "v":
                x = left + (width - box_w) * i / max(count - 1, 1) if count > 1 else left + (width - box_w) / 2
                y = top + (height - box_h) * layer / max(max_layer, 1) if max_layer else top + (height - box_h) / 2
            else:
                x = left + (width - box_w) * layer / max(max_layer, 1) if max_layer else left + (width - box_w) / 2
                y = top + (height - box_h) * i / max(count - 1, 1) if count > 1 else top + (height - box_h) / 2
            positions[nid] = (x, y)

    centers: Dict[str, tuple] = {}
    for nid, (x, y) in positions.items():
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, box_w, box_h)
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor.from_string(theme["accent_light"])
        shape.line.color.rgb = RGBColor.from_string(theme["accent"])
        shape.line.width = Pt(1.5)
        tf = shape.text_frame
        tf.word_wrap = True
        label = str(id_to_node[nid].get("label") or nid)
        _set_para(tf.paragraphs[0], label[:80], size=12, bold=True,
                  color="1F2937", font=theme["body_font"])
        centers[nid] = (x + box_w / 2, y + box_h / 2)

    for edge in edges:
        src, dst = str(edge[0]), str(edge[1])
        if src not in centers or dst not in centers:
            continue
        x1, y1 = centers[src]
        x2, y2 = centers[dst]
        conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Emu(int(x1)), Emu(int(y1)), Emu(int(x2)), Emu(int(y2)))
        conn.line.color.rgb = RGBColor.from_string(theme["accent_dark"])
        conn.line.width = Pt(1.5)
        try:
            _arrow(conn)
        except Exception:
            pass


def _build_pptx(title: str, spec: Dict[str, Any], theme: Dict[str, str]) -> bytes:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    slide_w, slide_h = prs.slide_width, prs.slide_height
    blank = prs.slide_layouts[6]
    accent = RGBColor.from_string(theme["accent"])
    accent_light = RGBColor.from_string(theme["accent_light"])

    for slide_spec in (spec.get("slides") or []):
        layout = str(slide_spec.get("layout") or "title_content").lower()
        s_title = str(slide_spec.get("title") or "").strip()
        slide = prs.slides.add_slide(blank)

        if layout == "title":
            band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(2.1), slide_w, Inches(2.9))
            band.fill.solid()
            band.fill.fore_color.rgb = accent
            band.line.fill.background()
            tf = _tb(slide, Inches(0.9), Inches(2.55), slide_w - Inches(1.8), Inches(1.6))
            p = tf.paragraphs[0]
            _set_para(p, s_title or title, size=40, bold=True, color="FFFFFF", font=theme["heading_font"])
            subtitle = str(slide_spec.get("body") or spec.get("subtitle") or "").strip()
            if subtitle:
                tf2 = _tb(slide, Inches(0.9), Inches(5.25), slide_w - Inches(1.8), Inches(1.0))
                _set_para(tf2.paragraphs[0], subtitle, size=18, bold=False,
                          color="4B5563", font=theme["body_font"])
        elif layout == "section":
            band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(2.9), slide_w, Inches(1.7))
            band.fill.solid()
            band.fill.fore_color.rgb = accent
            band.line.fill.background()
            tf = _tb(slide, Inches(0.9), Inches(3.15), slide_w - Inches(1.8), Inches(1.2))
            _set_para(tf.paragraphs[0], s_title, size=34, bold=True, color="FFFFFF", font=theme["heading_font"])
        elif layout == "closing":
            bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, slide_w, slide_h)
            bg.fill.solid()
            bg.fill.fore_color.rgb = accent_light
            bg.line.fill.background()
            tf = _tb(slide, Inches(1.0), Inches(2.6), slide_w - Inches(2.0), Inches(1.4))
            _set_para(tf.paragraphs[0], s_title or title, size=38, bold=True,
                      color=theme["accent_dark"], font=theme["heading_font"])
            body = str(slide_spec.get("body") or "").strip()
            if body:
                tf2 = _tb(slide, Inches(1.0), Inches(4.2), slide_w - Inches(2.0), Inches(2.0))
                _set_para(tf2.paragraphs[0], body, size=18, bold=False, color="374151", font=theme["body_font"])
        elif layout == "two_column":
            _title_header(slide, s_title, theme, slide_w)
            left_tf = _tb(slide, Inches(0.7), Inches(1.7), Inches(5.9), Inches(5.4))
            _fill_bullets(left_tf, slide_spec.get("bullets") or [], theme)
            right = str(slide_spec.get("body") or "")
            right_tf = _tb(slide, Inches(6.9), Inches(1.7), Inches(5.9), Inches(5.4))
            if right:
                _set_para(right_tf.paragraphs[0], right, size=16, bold=False,
                          color="1F2937", font=theme["body_font"])
        elif layout in ("chart", "chart_bullets"):
            _title_header(slide, s_title, theme, slide_w)
            chart = slide_spec.get("chart") or {}
            render = str(chart.get("render") or "native").lower()
            if layout == "chart":
                if render == "native":
                    _add_native_chart(slide, chart, theme, Inches(1.4), Inches(1.7), Inches(10.5), Inches(5.2))
                else:
                    _add_picture_chart(slide, chart, theme, Inches(2.6), Inches(1.7), Inches(8.0))
            else:
                if render == "native":
                    _add_native_chart(slide, chart, theme, Inches(0.5), Inches(1.7), Inches(7.3), Inches(5.2))
                else:
                    _add_picture_chart(slide, chart, theme, Inches(0.6), Inches(1.9), Inches(7.0))
                right_tf = _tb(slide, Inches(8.3), Inches(1.8), Inches(4.5), Inches(5.2))
                _fill_bullets(right_tf, slide_spec.get("bullets") or [], theme, base_size=15)
        elif layout == "diagram":
            _title_header(slide, s_title, theme, slide_w)
            _build_diagram(slide, slide_spec.get("diagram") or {}, theme,
                           Inches(0.7), Inches(1.7), slide_w - Inches(1.4), Inches(5.2))
        else:  # title_content fallback
            _title_header(slide, s_title, theme, slide_w)
            tf = _tb(slide, Inches(0.75), Inches(1.7), slide_w - Inches(1.5), Inches(5.4))
            _fill_bullets(tf, slide_spec.get("bullets") or [], theme)
            body = str(slide_spec.get("body") or "").strip()
            if body and not (slide_spec.get("bullets") or []):
                _set_para(tf.paragraphs[0], body, size=18, bold=False,
                          color="1F2937", font=theme["body_font"])
            elif body:
                p = tf.add_paragraph()
                _set_para(p, body, size=16, bold=False, color="374151", font=theme["body_font"])

        notes = str(slide_spec.get("notes") or "").strip()
        if notes:
            slide.notes_slide.notes_text_frame.text = notes

    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


# --------------------------------------------------------------------------
# xlsx / csv builders
# --------------------------------------------------------------------------

_INVALID_SHEET_CHARS = re.compile(r"[\[\]:*?/\\]")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?$")


def _sheet_name(value: Any, index: int, used: set) -> str:
    """Sanitize/dedupe an Excel sheet title (<=31 chars, no []:*?/\\)."""
    name = _INVALID_SHEET_CHARS.sub("", str(value or "").strip()) or f"Sheet{index + 1}"
    name = name[:31]
    candidate, n = name, 1
    while candidate.lower() in used:
        suffix = f" ({n})"
        candidate = name[:31 - len(suffix)] + suffix
        n += 1
    used.add(candidate.lower())
    return candidate


def _xlsx_cell(value: Any) -> tuple:
    """Return ``(cell_value, number_format_or_None)`` for a spec cell."""
    if value is None or isinstance(value, (int, float, bool)):
        return value, None
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("="):
            return s, None
        if _ISO_DATE_RE.match(s):
            try:
                parsed = datetime.datetime.fromisoformat(s)
                if "T" not in s and " " not in s:
                    return parsed.date(), "yyyy-mm-dd"
                return parsed, "yyyy-mm-dd hh:mm"
            except ValueError:
                pass
        return value, None
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False), None
    return str(value), None


def _column_format(fmt_map: Any, col_index: int, col_letter: str) -> Optional[str]:
    """Look up a per-column number format keyed by 1-based index or letter."""
    if isinstance(fmt_map, list):
        if 1 <= col_index <= len(fmt_map) and fmt_map[col_index - 1]:
            return str(fmt_map[col_index - 1])
        return None
    if isinstance(fmt_map, dict):
        for key in (col_index, str(col_index), col_letter, col_letter.lower()):
            if key in fmt_map and fmt_map[key]:
                return str(fmt_map[key])
    return None


def _add_xlsx_chart(wb, ws, chart_spec: Dict[str, Any], theme: Dict[str, str],
                    palette: List[str], n_cols: int, data_ws, data_col: int) -> tuple:
    """Render a sheet chart; returns updated ``(data_ws, next_data_col)``.

    Native openpyxl charts need cell references, so chart data is written to
    a hidden ``_chartdata`` worksheet. ``render='image'`` embeds the matplotlib
    PNG like docx charts instead.
    """
    from openpyxl.chart import AreaChart, BarChart, DoughnutChart, LineChart, PieChart, Reference
    from openpyxl.chart.label import DataLabelList
    from openpyxl.utils import get_column_letter

    series_list = chart_spec.get("series") or []
    if not series_list:
        return data_ws, data_col
    anchor = chart_spec.get("anchor") or f"{get_column_letter(n_cols + 2)}2"

    if (chart_spec.get("render") or "native") == "image":
        try:
            from openpyxl.drawing.image import Image as XLImage
            ws.add_image(XLImage(io.BytesIO(_render_chart_png(chart_spec, theme))), anchor)
        except Exception:
            pass
        return data_ws, data_col

    ctype = (chart_spec.get("type") or "column").lower()
    categories = chart_spec.get("categories") or []
    used_series = series_list[:1] if ctype in ("pie", "doughnut") else series_list

    if data_ws is None:
        data_ws = wb.create_sheet("_chartdata")
        data_ws.sheet_state = "hidden"
        data_col = 1
    base_col = data_col
    n_rows = max(len(categories), max(len(s.get("values") or []) for s in used_series))
    if categories:
        for r, cat in enumerate(categories, start=2):
            data_ws.cell(row=r, column=base_col, value=str(cat))
    first_val_col = base_col + (1 if categories else 0)
    for j, s in enumerate(used_series):
        col = first_val_col + j
        data_ws.cell(row=1, column=col, value=s.get("name") or f"Series {j + 1}")
        for r, v in enumerate(s.get("values") or [], start=2):
            data_ws.cell(row=r, column=col, value=v)

    if ctype in ("bar", "bar_stacked", "column", "column_stacked"):
        chart = BarChart()
        chart.type = "bar" if ctype.startswith("bar") else "col"
        if ctype.endswith("_stacked"):
            chart.grouping = "stacked"
            chart.overlap = 100
    elif ctype in ("line", "line_markers"):
        chart = LineChart()
    elif ctype == "pie":
        chart = PieChart()
    elif ctype == "doughnut":
        chart = DoughnutChart()
    elif ctype == "area":
        chart = AreaChart()
    else:
        return data_ws, base_col

    data_ref = Reference(data_ws, min_col=first_val_col, min_row=1,
                         max_col=first_val_col + len(used_series) - 1,
                         max_row=1 + n_rows)
    chart.add_data(data_ref, titles_from_data=True)
    if categories:
        cat_ref = Reference(data_ws, min_col=base_col, min_row=2, max_row=1 + len(categories))
        chart.set_categories(cat_ref)
    if chart_spec.get("title"):
        chart.title = chart_spec["title"]
    for ser, color in zip(chart.series, palette):
        ser.graphicalProperties.solidFill = color
        if ctype == "line_markers":
            ser.marker.symbol = "circle"
        if chart_spec.get("number_format"):
            try:
                ser.dLbls = DataLabelList()
                ser.dLbls.numFmt = chart_spec["number_format"]
                ser.dLbls.showVal = True
            except Exception:
                pass
    ws.add_chart(chart, anchor)
    return data_ws, first_val_col + len(used_series) + 1


def _build_xlsx(title: str, spec: Dict[str, Any], theme: Dict[str, str]) -> bytes:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError:
        raise RuntimeError("xlsx generation requires the 'openpyxl' package")

    wb = Workbook()
    wb.properties.title = title
    palette = _palette(theme["accent"])
    header_font = Font(bold=True, color="FFFFFF", name=theme["body_font"])
    header_fill = PatternFill("solid", fgColor=theme["accent"])
    used_names: set = set()
    data_ws, data_col = None, 0

    for i, sheet in enumerate(spec.get("sheets") or []):
        ws = wb.active if i == 0 else wb.create_sheet()
        ws.title = _sheet_name(sheet.get("name"), i, used_names)
        headers = sheet.get("headers") or []
        rows = sheet.get("rows") or []
        n_cols = max([len(headers)] + [len(r) for r in rows] + [1])

        if headers:
            for c, h in enumerate(headers, start=1):
                cell = ws.cell(row=1, column=c, value="" if h is None else h)
                cell.font = header_font
                cell.fill = header_fill
            if sheet.get("freeze_header", True):
                ws.freeze_panes = "A2"
            if sheet.get("auto_filter"):
                ws.auto_filter.ref = f"A1:{get_column_letter(n_cols)}{len(rows) + 1}"

        number_formats = sheet.get("number_formats")
        col_widths = sheet.get("column_widths")
        for r, row in enumerate(rows, start=2 if headers else 1):
            for c, raw in enumerate(row, start=1):
                value, fmt = _xlsx_cell(raw)
                if value is None:
                    continue
                cell = ws.cell(row=r, column=c, value=value)
                explicit = _column_format(number_formats, c, get_column_letter(c))
                if explicit or fmt:
                    cell.number_format = explicit or fmt

        for c in range(1, n_cols + 1):
            letter = get_column_letter(c)
            width = _column_format(col_widths, c, letter)
            if width is not None:
                try:
                    ws.column_dimensions[letter].width = max(4.0, min(255.0, float(width)))
                    continue
                except (TypeError, ValueError):
                    pass
            cells = [headers[c - 1] if c <= len(headers) else None] + [
                row[c - 1] for row in rows[:200] if c <= len(row)]
            longest = max((len(str(v)) for v in cells if v is not None), default=8)
            ws.column_dimensions[letter].width = max(8.0, min(50.0, longest + 2.0))

        chart = sheet.get("chart")
        if isinstance(chart, dict):
            data_ws, data_col = _add_xlsx_chart(
                wb, ws, chart, theme, palette, n_cols, data_ws, data_col)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _build_csv(spec: Dict[str, Any]) -> bytes:
    """Render the first sheet of the sheets spec as UTF-8 CSV (with BOM for Excel)."""
    import csv as _csv

    sheet = (spec.get("sheets") or [{}])[0] or {}
    buf = io.StringIO()
    writer = _csv.writer(buf)
    headers = sheet.get("headers") or []
    if headers:
        writer.writerow(["" if h is None else h for h in headers])
    for row in sheet.get("rows") or []:
        writer.writerow(["" if c is None else c for c in row])
    return buf.getvalue().encode("utf-8-sig")


# --------------------------------------------------------------------------
# tool
# --------------------------------------------------------------------------

_CHART_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": "Chart spec. Native editable chart for pptx/xlsx (default); render='image' for a styled image chart; docx charts are always images.",
    "properties": {
        "type": {
            "type": "string",
            "enum": ["bar", "bar_stacked", "column", "column_stacked", "line",
                     "line_markers", "pie", "doughnut", "area"],
            "default": "column",
        },
        "title": {"type": "string"},
        "categories": {"type": "array", "items": {"type": "string"}},
        "series": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "values": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["values"],
            },
        },
        "render": {"type": "string", "enum": ["native", "image"], "default": "native"},
        "number_format": {"type": "string", "description": "Data-label number format, e.g. '0.0%' or '#,##0'."},
        "anchor": {"type": "string", "description": "xlsx only: chart anchor cell, e.g. 'H2' (default: right of the data)."},
    },
    "required": ["series"],
}


class DocumentGeneratorTool:
    """Generate .docx/.pptx/.pdf/.xlsx/.csv documents from a structured spec."""

    name = "document_generator"
    description = (
        "Generate a downloadable file (.docx, .pptx, .pdf, .xlsx or .csv) from a "
        "structured spec. Use 'kind' and 'title'. "
        "'pdf' renders the docx 'sections' spec to a PDF file (requires LibreOffice). "
        "docx: 'sections' list with heading/level/paragraphs/bullets/table/chart. "
        "pptx: 'slides' list with layout (title, title_content, two_column, section, closing, "
        "chart, chart_bullets, diagram), title, bullets (strings or {text,level}), body, notes, "
        "chart {type, categories, series[{name,values}], render}, diagram {nodes, edges}. "
        "xlsx: 'sheets' list with name, headers, rows (list of row lists; numbers/booleans/"
        "ISO-dates are typed, strings starting with '=' become formulas), column_widths, "
        "freeze_header (default true), auto_filter, number_formats and an optional native "
        "Excel chart {type, categories, series, render, anchor}. "
        "csv: same 'sheets' spec — only the FIRST sheet is written (headers + rows), "
        "charts/styles are ignored. "
        "Native PowerPoint/Excel charts are editable; render='image' uses styled image charts. "
        "For small documents omit 'action' and pass sections/slides/sheets in one call. "
        "For long documents (>~15 sections or >~15 slides, many sheet rows, or very long "
        "text) build incrementally to keep each call small: action='create' (returns "
        "draft_id) -> repeated action='add_sections'/'add_slides'/'add_sheets' with "
        "draft_id (a few sections, ~10 slides or ~3 sheets per call) -> "
        "action='finalize' to produce the file; 'abort' discards. "
        "action='edit' rewrites draft items in place: 'edits' is an ordered batch of "
        "{operation: 'replace'|'insert'|'remove'|'move', index (0-based), "
        "from/to for move, section|slide|sheet payload for replace/insert} applied "
        "sequentially to the evolving item list — all-or-nothing (one failure "
        "leaves the draft unchanged), max 50 per call. "
        "Set 'workspace_path' to additionally save a copy into the agent workspace "
        "(requires workspace to be enabled). "
        "Delivered files appear as chat cards by default; pass deliver=false to keep "
        "the document internal, and action='publish' + filename to attach an existing "
        "artifact later. Reusing the same filename updates that artifact in place. "
        "Example docx section: {\"heading\": \"Ergebnisse\", \"level\": 1, "
        "\"paragraphs\": [\"Der Umsatz stieg.\"], \"bullets\": [\"Punkt A\"], "
        "\"table\": {\"headers\": [\"Kennzahl\", \"Wert\"], \"rows\": [[\"Umsatz\", 12.5]]}}. "
        "Pass content as structured fields — never embed JSON specs inside text fields."
    )

    def __init__(self, artifact_root: Path) -> None:
        self._store = ArtifactStore(Path(artifact_root))
        self._ws_store = AgentWorkspaceStore()

    @property
    def tool_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": ["docx", "pptx", "pdf", "xlsx", "csv"],
                            "description": "File type: 'docx' Word document, 'pptx' presentation, 'pdf' (sections spec, requires LibreOffice), 'xlsx' Excel workbook ('sheets' spec), 'csv' (first sheet of 'sheets' spec).",
                        },
                        "title": {
                            "type": "string",
                            "description": "Document title (title page / title slide).",
                        },
                        "filename": {
                            "type": "string",
                            "description": "Optional output filename; extension is forced to match 'kind'.",
                        },
                        "workspace_path": {
                            "type": "string",
                            "description": "Optional path relative to the agent workspace root (e.g. 'reports/q3.docx') — when set, a copy of the file is also written there (subdirectories are created). Extension is forced to match 'kind'. For drafts: pass at 'create' (stored) or 'finalize' (overrides).",
                        },
                        "author": {"type": "string", "description": "Document author metadata."},
                        "subject": {"type": "string", "description": "Document subject metadata."},
                        "action": {
                            "type": "string",
                            "enum": ["create", "add_sections", "add_slides", "add_sheets", "edit", "finalize", "abort", "publish"],
                            "description": "Draft lifecycle action. Omit for one-shot generation; 'create' starts a draft, 'add_sections'/'add_slides'/'add_sheets' append, 'edit' reorders/replaces items, 'finalize' builds, 'abort' discards. 'publish' requires 'filename' and attaches an existing artifact to the chat.",
                        },
                        "deliver": {
                            "type": "boolean",
                            "description": "Attach the finished file to the chat for user preview/download (default true). Set false to keep it internal; publish later via action='publish'. For drafts: pass at 'create' (stored) or 'finalize' (overrides).",
                        },
                        "draft_id": {
                            "type": "string",
                            "description": "Draft id returned by action='create' (required for add_sections/add_slides/add_sheets/finalize/abort).",
                        },
                        "sections": {
                            "type": "array",
                            "description": "docx only: content sections.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "heading": {"type": "string"},
                                    "level": {"type": "integer", "description": "Heading level 1-3 (default 1)."},
                                    "paragraphs": {"type": "array", "items": {"type": "string"}},
                                    "bullets": {"type": "array", "items": {"type": "string"}},
                                    "table": {
                                        "type": "object",
                                        "properties": {
                                            "headers": {"type": "array", "items": {"type": "string"}},
                                            "rows": {"type": "array", "items": {"type": "array"}},
                                        },
                                    },
                                    "chart": _CHART_SCHEMA,
                                },
                            },
                        },
                        "slides": {
                            "type": "array",
                            "description": "pptx only: slide specs.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "layout": {
                                        "type": "string",
                                        "enum": ["title", "title_content", "two_column", "section",
                                                 "closing", "chart", "chart_bullets", "diagram"],
                                        "default": "title_content",
                                    },
                                    "title": {"type": "string"},
                                    "bullets": {
                                        "type": "array",
                                        "description": "Strings or {text, level} items.",
                                        "items": {},
                                    },
                                    "body": {"type": "string", "description": "Free text block (subtitle on 'title'/'closing', right column on 'two_column')."},
                                    "notes": {"type": "string", "description": "Speaker notes."},
                                    "chart": _CHART_SCHEMA,
                                    "diagram": {
                                        "type": "object",
                                        "properties": {
                                            "nodes": {"type": "array", "items": {"type": "object", "properties": {"id": {}, "label": {"type": "string"}}, "required": ["id"]}},
                                            "edges": {"type": "array", "items": {"type": "array", "items": {}, "minItems": 2, "maxItems": 2}},
                                            "direction": {"type": "string", "enum": ["h", "v"], "default": "h"},
                                        },
                                        "required": ["nodes", "edges"],
                                    },
                                },
                            },
                        },
                        "sheets": {
                            "type": "array",
                            "description": "xlsx/csv only: worksheet specs. csv writes only the first sheet (headers + rows; chart/style fields ignored).",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string", "description": "Worksheet name (sanitized to <=31 chars)."},
                                    "headers": {"type": "array", "description": "Header row cells.", "items": {}},
                                    "rows": {"type": "array", "description": "Data rows — each row a list of cells. Numbers/booleans/ISO date strings are typed; strings starting with '=' become formulas.", "items": {"type": "array", "items": {}}},
                                    "column_widths": {"description": "Column widths keyed by 1-based column index or letter, or a list of widths."},
                                    "number_formats": {"description": "Number formats keyed by 1-based column index or letter, or a list (e.g. '#,##0.00', '0.0%')."},
                                    "freeze_header": {"type": "boolean", "default": True, "description": "Freeze the header row (default true)."},
                                    "auto_filter": {"type": "boolean", "default": False, "description": "Enable the Excel autofilter on the header row."},
                                    "chart": _CHART_SCHEMA,
                                },
                            },
                        },
                        "edits": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": (
                                "For action='edit': ordered batch of draft item ops "
                                "(max 50, all-or-nothing). Each: {operation: 'replace'|"
                                "'insert'|'remove'|'move'} — 'index' (0-based position "
                                "in the current item list; insert allows 0..len), "
                                "'from'/'to' for move, and a 'section' (docx/pdf drafts), "
                                "'slide' (pptx drafts) or 'sheet' (xlsx/csv drafts) "
                                "payload object for "
                                "replace/insert. Indexes apply to the evolving list — "
                                "later ops see earlier results."
                            ),
                        },
                    },
                    "required": [],
                },
            },
        }

    def call_conflict_key(self, args: Dict[str, Any]) -> Optional[str]:
        """Serialize same-draft mutations inside a parallel tool-call batch."""
        draft_id = args.get("draft_id")
        if draft_id:
            return f"draft:{draft_id}"
        ws = str(args.get("workspace_path") or "").strip().replace("\\", "/")
        if ws:
            return f"ws:{ws}"
        return None

    async def execute(
        self,
        kind: Optional[str] = None,
        title: Optional[str] = None,
        filename: Optional[str] = None,
        workspace_path: Optional[str] = None,
        author: Optional[str] = None,
        subject: Optional[str] = None,
        sections: Optional[List[Dict[str, Any]]] = None,
        slides: Optional[List[Dict[str, Any]]] = None,
        sheets: Optional[List[Dict[str, Any]]] = None,
        action: Optional[str] = None,
        draft_id: Optional[str] = None,
        edits: Optional[List[Dict[str, Any]]] = None,
        deliver: Optional[bool] = None,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ctx = context or {}
        conversation_id = ctx.get("conversation_id")
        if not conversation_id:
            return _err("conversation_id required for document_generator")

        action = (action or "").lower().strip()
        if action in ("create", "add_sections", "add_slides", "add_sheets",
                      "edit", "finalize", "abort", "publish"):
            return await self._execute_action(
                action, conversation_id, ctx,
                kind=kind, title=title, filename=filename, author=author,
                subject=subject, sections=sections, slides=slides,
                sheets=sheets, draft_id=draft_id, edits=edits,
                workspace_path=workspace_path, deliver=deliver,
            )
        if action:
            return _err(
                f"unknown action '{action}'; use create|add_sections|add_slides|add_sheets|"
                "edit|finalize|abort|publish, or omit 'action' for one-shot generation"
            )

        kind = (kind or "").lower().strip()
        if kind not in _VALID_KINDS:
            return _err("kind must be 'docx', 'pptx', 'pdf', 'xlsx' or 'csv'")
        title = (title or "").strip()
        if not title:
            return _err("title is required")

        spec = {
            "sections": sections or [],
            "slides": slides or [],
            "sheets": sheets or [],
            "author": author,
            "subject": subject,
        }
        # 'pdf' renders the docx sections spec — normalize/validate as docx.
        norm_error = _normalize_items(spec, _spec_kind(kind))
        if norm_error:
            return _err(norm_error)
        return await self._finish(
            conversation_id, ctx, kind, title, filename, spec,
            workspace_path=workspace_path, deliver=deliver is not False,
        )

    def _resolve_workspace_rel(
        self, ctx: Dict[str, Any], workspace_path: Optional[str], kind: str
    ) -> tuple:
        """Normalize and validate a ``workspace_path`` arg.

        Returns ``(rel_path, error)`` — ``rel_path`` is ``None`` when unset or
        invalid; ``error`` is ``None`` when the path is usable.
        """
        ws_rel = (workspace_path or "").strip().replace("\\", "/")
        if not ws_rel:
            return None, None
        scope_dir = ctx.get("agent_workspace_dir")
        if not scope_dir:
            return None, "workspace_path given but agent workspace is not enabled"
        # Force the extension on the basename, same rule as out_name.
        stem = Path(ws_rel).stem
        parent = str(Path(ws_rel).parent)
        ws_rel = f"{parent}/{stem}.{kind}" if parent not in (".", "") else f"{stem}.{kind}"
        try:
            self._ws_store.resolve(Path(scope_dir), ws_rel)
        except Exception as exc:
            return None, f"invalid workspace_path: {exc}"
        return ws_rel, None

    async def _finish(
        self,
        conversation_id: str,
        ctx: Dict[str, Any],
        kind: str,
        title: str,
        filename: Optional[str],
        spec: Dict[str, Any],
        workspace_path: Optional[str] = None,
        deliver: bool = True,
    ) -> Dict[str, Any]:
        spec_kind = _spec_kind(kind)
        validation = self._validate(spec, spec_kind)
        if validation is not None:
            return _err(validation)

        ws_rel, ws_err = self._resolve_workspace_rel(ctx, workspace_path, kind)
        if ws_err:
            return _err(ws_err)

        theme = _theme(ctx)
        try:
            if kind == "csv":
                data = await asyncio.to_thread(_build_csv, spec)
            else:
                builder = {"docx": _build_docx, "pptx": _build_pptx,
                           "xlsx": _build_xlsx}[spec_kind]
                data = await asyncio.to_thread(builder, title, spec, theme)
        except Exception as exc:
            logger.warning("document_generator build failed: %s", exc, exc_info=True)
            return _err(f"Failed to build document: {exc}")

        if kind == "pdf":
            converter = self._store.converter
            if not converter.available:
                return _err("PDF generation requires LibreOffice on the server.")
            try:
                import tempfile
                with tempfile.TemporaryDirectory() as tmpdir:
                    src = Path(tmpdir) / "document.docx"
                    await asyncio.to_thread(src.write_bytes, data)
                    pdf_path = await converter.convert(src, "pdf", Path(tmpdir))
                    data = await asyncio.to_thread(pdf_path.read_bytes)
            except Exception as exc:
                logger.warning("document_generator pdf conversion failed: %s", exc, exc_info=True)
                return _err(f"Failed to convert document to PDF: {exc}")

        out_name = filename or f"{_slug(title)}.{kind}"
        if not out_name.lower().endswith(f".{kind}"):
            out_name = f"{Path(out_name).stem}.{kind}"
        model = str(ctx.get("assistant_name") or "")
        try:
            entry = await self._store.save(
                conversation_id, model, out_name, data, kind, overwrite=True)
        except Exception as exc:
            logger.warning("document_generator save failed: %s", exc, exc_info=True)
            return _err(f"Failed to store document: {exc}")

        if spec_kind == "pptx":
            detail = f"{len(spec['slides'])} slide{'s' if len(spec['slides']) != 1 else ''}"
        elif spec_kind == "xlsx":
            if kind == "csv":
                n_rows = len((spec["sheets"][0] or {}).get("rows") or []) if spec["sheets"] else 0
                detail = f"{n_rows} row{'s' if n_rows != 1 else ''}"
            else:
                n = len(spec["sheets"])
                detail = f"{n} sheet{'s' if n != 1 else ''}"
        else:
            detail = f"{len(spec['sections'])} section{'s' if len(spec['sections']) != 1 else ''}"
        ws_note = ""
        ws_meta: Optional[str] = None
        if ws_rel:
            try:
                ws_cfg = ctx.get("agent_workspace_cfg") or {}
                ws_entry = await asyncio.to_thread(
                    self._ws_store.write, Path(ctx["agent_workspace_dir"]), ws_rel, data, ws_cfg)
                ws_meta = ws_entry["path"]
                ws_note = f" Also saved to the workspace at '{ws_entry['path']}'."
            except Exception as exc:
                logger.warning("document_generator workspace copy failed: %s", exc, exc_info=True)
                ws_note = f" Workspace copy failed: {exc}"
        if deliver:
            msg = f"Created '{entry['name']}' — {detail}, {entry['size'] / 1024:.0f} KB.{ws_note} The user can download and preview it in the chat."
        else:
            msg = (f"Created '{entry['name']}' — {detail}, {entry['size'] / 1024:.0f} KB."
                   f"{ws_note} Saved internally — not shown in the chat; call action='publish' "
                   "with this filename to attach it later.")
        metadata: Dict[str, Any] = {}
        if deliver:
            metadata["files"] = [entry]
        if ws_meta:
            metadata["workspace_path"] = ws_meta
        return {"ok": True, "content": msg, "metadata": metadata}

    # -- incremental drafts --------------------------------------------------

    def _draft_path(self, conversation_id: str, draft_id: str) -> Optional[Path]:
        if not re.fullmatch(r"doc-[0-9a-f]+", draft_id or ""):
            return None
        safe = sanitize_filename(draft_id)
        if safe != draft_id:
            return None
        return self._store.draft_dir(conversation_id) / f"{draft_id}.json"

    async def _load_draft(self, conversation_id: str, draft_id: Optional[str]) -> tuple:
        path = self._draft_path(conversation_id, draft_id or "")
        if path is None or not path.is_file():
            return None, f"draft '{draft_id}' not found — call action='create' first"
        try:
            draft = json.loads(await asyncio.to_thread(path.read_text, encoding="utf-8"))
        except Exception:
            return None, f"draft '{draft_id}' is unreadable"
        for container in ("sections", "slides", "sheets"):
            draft.setdefault(container, [])
        if time.time() - float(draft.get("created_at") or 0) > _DRAFT_TTL_SECONDS:
            await asyncio.to_thread(path.unlink, True)
            return None, f"draft '{draft_id}' expired — call action='create' again"
        return draft, None

    async def _save_draft(self, conversation_id: str, draft: Dict[str, Any]) -> Path:
        draft_dir = self._store.draft_dir(conversation_id)
        await asyncio.to_thread(draft_dir.mkdir, parents=True, exist_ok=True)
        path = draft_dir / f"{draft['draft_id']}.json"
        tmp = path.with_suffix(".tmp")
        await asyncio.to_thread(tmp.write_text, json.dumps(draft, ensure_ascii=False), "utf-8")
        await asyncio.to_thread(os.replace, tmp, path)
        return path

    async def _delete_draft(self, conversation_id: str, draft_id: str) -> None:
        path = self._draft_path(conversation_id, draft_id)
        if path is not None:
            await asyncio.to_thread(path.unlink, True)

    async def _execute_action(
        self,
        action: str,
        conversation_id: str,
        ctx: Dict[str, Any],
        *,
        kind: Optional[str],
        title: Optional[str],
        filename: Optional[str],
        author: Optional[str],
        subject: Optional[str],
        sections: Optional[List[Dict[str, Any]]],
        slides: Optional[List[Dict[str, Any]]],
        sheets: Optional[List[Dict[str, Any]]] = None,
        draft_id: Optional[str] = None,
        edits: Optional[List[Dict[str, Any]]] = None,
        workspace_path: Optional[str] = None,
        deliver: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if action == "publish":
            name = (filename or "").strip()
            if not name:
                return _err("action 'publish' requires 'filename'")
            entry = self._store.get_entry(conversation_id, name)
            if entry is None:
                return _err(f"no artifact named '{name}' in this conversation")
            file_entry = {
                "name": entry["name"],
                "conversation_id": conversation_id,
                "kind": entry["kind"],
                "mime": entry["mime"],
                "size": entry["size"],
                "updated_at": entry.get("updated_at"),
            }
            return {"ok": True,
                    "content": f"Published '{entry['name']}' to the chat.",
                    "metadata": {"files": [file_entry]}}

        if action == "create":
            kind = (kind or "").lower().strip()
            if kind not in _VALID_KINDS:
                return _err("action 'create' requires 'kind' ('docx', 'pptx', 'pdf', 'xlsx' or 'csv')")
            title = (title or "").strip()
            if not title:
                return _err("action 'create' requires 'title'")
            if workspace_path:
                _, ws_err = self._resolve_workspace_rel(ctx, workspace_path, kind)
                if ws_err:
                    return _err(ws_err)
            draft = {
                "draft_id": f"doc-{uuid.uuid4().hex[:12]}",
                "kind": kind,
                "title": title,
                "filename": filename,
                "workspace_path": workspace_path,
                "author": author,
                "subject": subject,
                "deliver": True if deliver is None else bool(deliver),
                "sections": [],
                "slides": [],
                "sheets": [],
                "created_at": time.time(),
            }
            if sections or slides or sheets:
                appended = self._append_items(
                    draft, _spec_kind(kind),
                    sections or [], slides or [], sheets or [])
                if isinstance(appended, str):
                    return _err(appended)
            await self._save_draft(conversation_id, draft)
            return {"ok": True, "content": (
                f"Draft created ({kind}, draft_id='{draft['draft_id']}'). "
                f"Append content with action='add_sections'/'add_slides'/'add_sheets' + draft_id, "
                f"then call action='finalize' to build the file."
            )}

        draft, error = await self._load_draft(conversation_id, draft_id)
        if error:
            return _err(error)

        if action == "abort":
            await self._delete_draft(conversation_id, draft["draft_id"])
            return {"ok": True, "content": f"Draft '{draft['draft_id']}' discarded."}

        if action == "finalize":
            spec = {
                "sections": draft["sections"],
                "slides": draft["slides"],
                "sheets": draft["sheets"],
                "author": draft.get("author"),
                "subject": draft.get("subject"),
            }
            result = await self._finish(
                conversation_id, ctx, draft["kind"], draft["title"],
                draft.get("filename"), spec,
                workspace_path=(
                    workspace_path if workspace_path is not None
                    else draft.get("workspace_path")
                ),
                deliver=(deliver if deliver is not None
                         else bool(draft.get("deliver", True))),
            )
            if result.get("ok"):
                await self._delete_draft(conversation_id, draft["draft_id"])
            return result

        if action == "edit":
            error, applied = self._apply_draft_edits(draft, edits)
            if error:
                return _err(error)
            draft["created_at"] = time.time()
            await self._save_draft(conversation_id, draft)
            dkind = _spec_kind(draft["kind"])
            container = _CONTAINER_FOR_KIND[dkind]
            return {"ok": True, "content": (
                f"Applied {len(applied)} edit(s) to draft '{draft['draft_id']}' "
                f"({', '.join(applied)}) — now has {len(draft[container])} "
                f"{container}. Call action='finalize' when done."
            )}

        # add_sections / add_slides / add_sheets — 'pdf' drafts are document
        # drafts (sections), 'csv' drafts are sheets drafts.
        dkind = _spec_kind(draft["kind"])
        expected_action = f"add_{_CONTAINER_FOR_KIND[dkind]}"
        if action != expected_action:
            return _err(
                f"this is a '{draft['kind']}' draft — use action='{expected_action}'")
        error = self._append_items(
            draft, dkind, sections or [], slides or [], sheets or [])
        if isinstance(error, str):
            return _err(error)
        draft["created_at"] = time.time()
        await self._save_draft(conversation_id, draft)
        container = _CONTAINER_FOR_KIND[dkind]
        cap = {"sections": _MAX_SECTIONS, "slides": _MAX_SLIDES,
               "sheets": _MAX_SHEETS}[container]
        appended = {"sections": sections, "slides": slides,
                    "sheets": sheets}[container] or []
        return {"ok": True, "content": (
            f"Appended {len(appended)} {container} — draft now has "
            f"{len(draft[container])}/{cap} {container}. "
            "Call action='finalize' when done."
        )}

    def _apply_draft_edits(
        self,
        draft: Dict[str, Any],
        edits: Optional[List[Dict[str, Any]]],
    ) -> tuple:
        """Apply an ordered replace/insert/remove/move batch to draft items.

        Ops run sequentially on the evolving list; indexes are 0-based and
        refer to the state left by previous ops. Any failure restores the
        original item list so the saved draft is untouched (all-or-nothing).
        Returns ``(error, applied_labels)`` — ``error`` is None on success.
        """
        from agent.tools.builtin_tools.common import (
            coerce_edit_args,
            validate_edits_list,
        )

        try:
            ops = validate_edits_list(edits)
        except ValueError as exc:
            return str(exc), []

        dkind = _spec_kind(draft["kind"])
        container = _CONTAINER_FOR_KIND[dkind]
        payload_key = _PAYLOAD_KEY_FOR_KIND[dkind]
        items = draft[container]
        original = list(items)
        labels: List[str] = []

        def _normalized_payload(i: int, op_label: str) -> tuple:
            payload = ops[i].get(payload_key)
            if payload is None:
                raise ValueError(f"'{op_label}' requires a '{payload_key}' object")
            spec = {container: [payload]}
            norm_error = _normalize_items(spec, dkind)
            if norm_error:
                raise ValueError(norm_error)
            return spec[container][0]

        for i, raw in enumerate(ops):
            op_label = "?"
            try:
                op = coerce_edit_args(raw)
                op_label = str(op.get("operation") or "").strip().lower() or "(missing)"
                n = len(items)
                if op_label == "replace":
                    idx = op.get("index")
                    if idx is None or not (0 <= idx < n):
                        raise ValueError(
                            f"replace requires 'index' in 0..{max(n - 1, 0)} "
                            f"({n} item(s))"
                        )
                    items[idx] = _normalized_payload(i, "replace")
                    labels.append(f"replace@{idx}")
                elif op_label == "insert":
                    idx = op.get("index")
                    if idx is None or not (0 <= idx <= n):
                        raise ValueError(
                            f"insert requires 'index' in 0..{n} ({n} item(s))"
                        )
                    items.insert(idx, _normalized_payload(i, "insert"))
                    labels.append(f"insert@{idx}")
                elif op_label == "remove":
                    idx = op.get("index")
                    if idx is None or not (0 <= idx < n):
                        raise ValueError(
                            f"remove requires 'index' in 0..{max(n - 1, 0)} "
                            f"({n} item(s))"
                        )
                    items.pop(idx)
                    labels.append(f"remove@{idx}")
                elif op_label == "move":
                    src, dst = op.get("from"), op.get("to")
                    if (
                        src is None or dst is None
                        or not (0 <= src < n) or not (0 <= dst < n)
                    ):
                        raise ValueError(
                            f"move requires 'from'/'to' in 0..{max(n - 1, 0)} "
                            f"({n} item(s))"
                        )
                    if src != dst:
                        items.insert(dst, items.pop(src))
                    labels.append(f"move {src}→{dst}")
                else:
                    raise ValueError(
                        f"unknown operation '{op_label}' — use replace|insert|remove|move"
                    )
            except (ValueError, TypeError) as exc:
                draft[container] = original
                return (
                    f"edit #{i} ({op_label}) failed: {exc} — draft unchanged",
                    [],
                )

        cap_error = self._validate(
            {"sections": draft["sections"], "slides": draft["slides"],
             "sheets": draft.get("sheets") or [], "author": None, "subject": None},
            dkind,
            allow_empty=True,
        )
        if cap_error:
            draft[container] = original
            return cap_error + " — draft unchanged", []
        return None, labels

    def _append_items(
        self,
        draft: Dict[str, Any],
        kind: str,
        sections: List[Dict[str, Any]],
        slides: List[Dict[str, Any]],
        sheets: List[Dict[str, Any]],
    ) -> Optional[str]:
        spec = {"sections": sections, "slides": slides, "sheets": sheets}
        norm_error = _normalize_items(spec, kind)
        if norm_error:
            return norm_error
        container = _CONTAINER_FOR_KIND[kind]
        wrong = [c for c in spec if c != container and spec[c]]
        if wrong:
            return f"this draft accepts '{container}', not '{wrong[0]}'"
        if not spec[container]:
            return f"'add_{container}' requires a non-empty '{container}' list"

        draft[container].extend(spec[container])
        cap_error = self._validate(
            {"sections": draft["sections"], "slides": draft["slides"],
             "sheets": draft["sheets"], "author": None, "subject": None},
            kind,
            allow_empty=True,
        )
        if cap_error:
            del draft[container][-len(spec[container]):]
            return cap_error + " — send a smaller batch or finalize now"
        return None

    def _validate(self, spec: Dict[str, Any], kind: str, allow_empty: bool = False) -> Optional[str]:
        sections, slides = spec["sections"], spec["slides"]
        sheets = spec.get("sheets") or []
        if not allow_empty and kind == "docx" and not sections:
            return "docx requires a non-empty 'sections' list"
        if not allow_empty and kind == "pptx" and not slides:
            return "pptx requires a non-empty 'slides' list"
        if not allow_empty and kind == "xlsx" and not sheets:
            return "sheets spec requires a non-empty 'sheets' list"
        if len(sections) > _MAX_SECTIONS:
            return f"too many sections (max {_MAX_SECTIONS})" + _INCREMENTAL_HINT
        if len(slides) > _MAX_SLIDES:
            return f"too many slides (max {_MAX_SLIDES})" + _INCREMENTAL_HINT
        if len(sheets) > _MAX_SHEETS:
            return f"too many sheets (max {_MAX_SHEETS})" + _INCREMENTAL_HINT

        charts = 0
        text_chars = 0
        for i, s in enumerate(sections):
            if not isinstance(s, dict):
                return f"sections[{i}] must be an object"
            if not allow_empty and not any(s.get(k) for k in ("heading", "paragraphs", "bullets", "table", "chart")):
                return (f"sections[{i}] has no content — needs heading, paragraphs, "
                        "bullets, table or chart")
            text_chars += sum(len(str(p)) for p in s.get("paragraphs") or [])
            text_chars += sum(len(str(b)) for b in s.get("bullets") or [])
            if s.get("chart"):
                charts += 1
        for i, s in enumerate(slides):
            if not isinstance(s, dict):
                return f"slides[{i}] must be an object"
            if not allow_empty and not any(s.get(k) for k in ("title", "bullets", "body", "notes", "chart", "diagram")):
                return (f"slides[{i}] has no content — needs title, bullets, body, "
                        "notes, chart or diagram")
            text_chars += sum(len(str(b if isinstance(b, str) else b.get("text", ""))) for b in s.get("bullets") or [])
            text_chars += len(str(s.get("body") or "")) + len(str(s.get("notes") or ""))
            if s.get("chart"):
                charts += 1
            if len((s.get("diagram") or {}).get("nodes") or []) > _MAX_NODES:
                return f"diagram has too many nodes (max {_MAX_NODES})"
        total_cells = 0
        for i, s in enumerate(sheets):
            if not isinstance(s, dict):
                return f"sheets[{i}] must be an object"
            rows = s.get("rows") or []
            n_cols = max([len(s.get("headers") or [])]
                         + [len(r) for r in rows] + [0])
            if not s.get("headers") and not rows:
                return f"sheets[{i}] requires 'headers' or 'rows'"
            if len(rows) > _MAX_ROWS:
                return f"sheets[{i}] has too many rows (max {_MAX_ROWS})" + _INCREMENTAL_HINT
            if n_cols > _MAX_COLUMNS:
                return f"sheets[{i}] has too many columns (max {_MAX_COLUMNS})"
            total_cells += len(rows) * n_cols
            text_chars += sum(len(str(c)) for r in rows for c in r)
            if s.get("chart"):
                charts += 1
        if total_cells > _MAX_TOTAL_CELLS:
            return f"too many cells across all sheets (max {_MAX_TOTAL_CELLS})" + _INCREMENTAL_HINT
        if charts > _MAX_CHARTS:
            return f"too many charts (max {_MAX_CHARTS})" + _INCREMENTAL_HINT
        if text_chars > _MAX_TEXT_CHARS:
            return f"content too large (max {_MAX_TEXT_CHARS} chars)" + _INCREMENTAL_HINT
        for container in sections + slides + sheets:
            chart = container.get("chart") or {}
            if chart and not (chart.get("series") or []):
                return "chart requires a non-empty 'series' list"
            if chart and any(
                not isinstance(v, (int, float))
                for s in chart.get("series") or []
                for v in s.get("values") or []
            ):
                return "chart series values must be numbers"
        return None
