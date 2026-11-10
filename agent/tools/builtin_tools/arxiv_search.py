"""arXiv search builtin tool."""

from __future__ import annotations

import logging
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

import aiohttp

logger = logging.getLogger(__name__)


class ArxivSearchTool:
    """Search arXiv for academic papers via the Atom API."""

    name = "arxiv_search"
    description = (
        "Search arXiv for academic papers. No API key required. "
        "Use 'query' (required), optional 'max_results' (default 5, max 10), "
        "'sort_by' ('relevance', 'lastUpdatedDate', 'submittedDate'), and "
        "'sort_order' ('ascending', 'descending')."
    )

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
                        "query": {
                            "type": "string",
                            "description": "Search query for arXiv papers.",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results (default 5, max 10).",
                            "default": 5,
                        },
                        "sort_by": {
                            "type": "string",
                            "enum": ["relevance", "lastUpdatedDate", "submittedDate"],
                            "description": "Sort criterion.",
                            "default": "relevance",
                        },
                        "sort_order": {
                            "type": "string",
                            "enum": ["ascending", "descending"],
                            "description": "Sort order.",
                            "default": "descending",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    async def execute(
        self,
        query: str,
        max_results: int = 5,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> str:
        if max_results > 10:
            max_results = 10

        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }
        url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
        timeout = aiohttp.ClientTimeout(total=60)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"arXiv API returned HTTP {resp.status}")
                xml_text = await resp.text()

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            raise RuntimeError(f"Failed to parse arXiv response: {exc}")

        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        entries = root.findall("atom:entry", ns)
        if not entries:
            return f"No results found for: {query}"

        lines = [f"arXiv search results for: {query}", ""]
        for idx, entry in enumerate(entries, 1):
            title = entry.findtext("atom:title", "", ns).strip()
            summary = entry.findtext("atom:summary", "", ns).strip()
            id_url = entry.findtext("atom:id", "", ns).strip()
            published = entry.findtext("atom:published", "", ns).strip()
            updated = entry.findtext("atom:updated", "", ns).strip()

            authors = [a.findtext("atom:name", "", ns) for a in entry.findall("atom:author", ns)]

            pdf_link = ""
            for link in entry.findall("atom:link", ns):
                if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                    pdf_link = link.get("href", "")
                    break

            lines.append(f"{idx}. {title}")
            if authors:
                lines.append(f"   Authors: {', '.join(authors)}")
            if published:
                lines.append(f"   Published: {published}")
            if updated and updated != published:
                lines.append(f"   Updated: {updated}")
            if id_url:
                lines.append(f"   arXiv ID: {id_url}")
            if pdf_link:
                lines.append(f"   PDF: {pdf_link}")
            if summary:
                truncated = summary[:500] + "..." if len(summary) > 500 else summary
                lines.append(f"   Summary: {truncated}")
            lines.append("")

        return "\n".join(lines)
