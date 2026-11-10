"""Convert downloaded documents into Google Open Knowledge Format (OKF) Concepts.

OKF Concepts are markdown documents with YAML frontmatter. This module
wraps existing Document objects (produced by KnowledgeLoader) with rich
metadata so they can be stored and retrieved more effectively.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from langchain.schema import Document

# Mapping from file extension to OKF concept type
_EXTENSION_TO_TYPE: Dict[str, str] = {
    ".pdf": "document",
    ".txt": "text",
    ".md": "text",
    ".doc": "document",
    ".docx": "document",
    ".odt": "document",
    ".xls": "spreadsheet",
    ".xlsx": "spreadsheet",
    ".ppt": "presentation",
    ".pptx": "presentation",
    ".csv": "spreadsheet",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".gif": "image",
    ".bmp": "image",
    ".tiff": "image",
    ".webp": "image",
}


def _okf_type_from_extension(href: str) -> str:
    suffix = Path(href).suffix.lower()
    return _EXTENSION_TO_TYPE.get(suffix, "text")


def _truncate_text(text: str, max_len: int = 500) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rsplit(" ", 1)[0] + "..."


class OKFConverter:
    """Convert a Document into an OKF-style Concept Document.

    The returned Document has:
      - page_content: the original body text (no YAML frontmatter)
      - metadata: enriched with okf_* keys and the YAML frontmatter dict
    """

    @staticmethod
    def convert_document(
        doc: Document,
        source_href: str,
        meta: Dict[str, Any],
    ) -> Document:
        """Wrap a Document with OKF metadata.

        Args:
            doc: Original Document from KnowledgeLoader.
            source_href: WebDAV href (used as the source identifier).
            meta: WebDAV entry metadata dict with keys like
                  etag, content_length, last_modified.
        """
        page_content = doc.page_content or ""
        okf_type = _okf_type_from_extension(source_href)

        # Derive a title: use existing metadata title, or filename, or first line
        existing_title = str(doc.metadata.get("title") or "") if doc.metadata else ""
        if not existing_title:
            existing_title = Path(source_href).name

        description = _truncate_text(page_content)

        frontmatter: Dict[str, Any] = {
            "title": existing_title,
            "type": okf_type,
            "description": description,
            "source": source_href,
            "date_downloaded": datetime.now(timezone.utc).isoformat(),
            "file_size": meta.get("content_length"),
            "etag": meta.get("etag", ""),
        }

        # Clean None values from frontmatter
        frontmatter = {k: v for k, v in frontmatter.items() if v is not None}

        # Build enriched metadata
        new_metadata = dict(doc.metadata) if doc.metadata else {}
        new_metadata.update({
            "okf_concept_id": source_href,
            "okf_type": okf_type,
            "okf_title": existing_title,
            "okf_frontmatter": frontmatter,
            "source": source_href,
        })

        return Document(
            page_content=page_content,
            metadata=new_metadata,
        )
