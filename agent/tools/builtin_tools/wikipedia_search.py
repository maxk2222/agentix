"""Wikipedia search builtin tool."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

from .common import read_partial_text

logger = logging.getLogger(__name__)

_USER_AGENT = "Agentix-Agent/1.0 (educational RAG assistant; contact: admin@localhost)"
_HEADERS = {"User-Agent": _USER_AGENT}

_READ_LIMIT_DEFAULT = 50_000
_READ_LIMIT_MAX = 250_000


def _fetch_json(base_url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    query = urllib.parse.urlencode(params)
    url = f"{base_url}?{query}"
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


class WikipediaSearchTool:
    """Search Wikipedia and read full articles via the MediaWiki API."""

    name = "wikipedia_search"
    description = (
        "Search Wikipedia for articles or read a full article. No API key required. "
        "action='search' (default): 'query' (required), optional 'language' (default 'en'), "
        "'limit' (default 5, max 10), 'fetch_extracts' (default true). "
        "action='read': 'title' (required, falls back to 'query') returns the complete "
        "plain-text article with '== Section ==' markers, paged via 'read_limit' "
        "(default 50000, max 250000), 'offset', 'lines_start'/'lines_end' and 'pattern'."
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
                        "action": {
                            "type": "string",
                            "enum": ["search", "read"],
                            "description": "'search' finds articles; 'read' returns the full plain-text article for 'title'.",
                            "default": "search",
                        },
                        "query": {
                            "type": "string",
                            "description": "Search query (required for action='search'; fallback title for action='read').",
                        },
                        "title": {
                            "type": "string",
                            "description": "Article title to read in full (required for action='read').",
                        },
                        "language": {
                            "type": "string",
                            "description": "Wikipedia language code (default 'en').",
                            "default": "en",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of search results (default 5, max 10).",
                            "default": 5,
                        },
                        "fetch_extracts": {
                            "type": "boolean",
                            "description": "Fetch 3-sentence article extracts in search results (default true).",
                            "default": True,
                        },
                        "read_limit": {
                            "type": "integer",
                            "description": "Max characters of article text returned per read call (default 50000, max 250000). Page further with offset/lines_start.",
                            "default": _READ_LIMIT_DEFAULT,
                        },
                        "offset": {
                            "type": "integer",
                            "description": "action='read': character offset into the article text (negative counts from the end).",
                        },
                        "lines_start": {
                            "type": "integer",
                            "description": "action='read': first line to return (1-based, takes priority over offset).",
                        },
                        "lines_end": {
                            "type": "integer",
                            "description": "action='read': last line to return (inclusive).",
                        },
                        "pattern": {
                            "type": "string",
                            "description": "action='read': case-insensitive substring filter — returns numbered match windows instead of a range.",
                        },
                        "context_lines": {
                            "type": "integer",
                            "description": "action='read': lines of context around each pattern match (default 2).",
                        },
                    },
                    "required": [],
                },
            },
        }

    async def execute(
        self,
        query: Optional[str] = None,
        language: str = "en",
        limit: int = 5,
        fetch_extracts: bool = True,
        action: str = "search",
        title: Optional[str] = None,
        read_limit: int = _READ_LIMIT_DEFAULT,
        offset: Optional[int] = None,
        lines_start: Optional[int] = None,
        lines_end: Optional[int] = None,
        pattern: Optional[str] = None,
        context_lines: Optional[int] = None,
    ) -> str:
        if not re.match(r"^[a-z]{2,3}$", language or ""):
            raise ValueError(f"Invalid language code: {language}")
        action = (action or "search").lower().strip()
        base_url = f"https://{language}.wikipedia.org/w/api.php"

        if action == "read":
            return await self._execute_read(
                base_url,
                language,
                (title or "").strip() or query,
                read_limit=read_limit,
                offset=offset,
                lines_start=lines_start,
                lines_end=lines_end,
                pattern=pattern,
                context_lines=context_lines,
            )
        if action != "search":
            return f"Unknown action '{action}'. Use 'search' or 'read'."

        if not (query or "").strip():
            return "A query is required for action='search'."
        if limit > 10:
            limit = 10
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
            "origin": "*",
        }

        data = await asyncio.to_thread(_fetch_json, base_url, search_params)

        search_results = data.get("query", {}).get("search", [])
        if not search_results:
            return f"No Wikipedia results found for: {query}"

        lines = [f"Wikipedia search results for: {query}", ""]

        for idx, item in enumerate(search_results, 1):
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            page_id = item.get("pageid", "")
            url = f"https://{language}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"

            lines.append(f"{idx}. {title}")
            lines.append(f"   URL: {url}")
            if snippet:
                clean_snippet = re.sub(r"<[^>]+>", "", snippet)
                lines.append(f"   {clean_snippet}")

            if fetch_extracts and page_id:
                extract_params = {
                    "action": "query",
                    "prop": "extracts",
                    "pageids": page_id,
                    "explaintext": "true",
                    "exsentences": 3,
                    "format": "json",
                    "origin": "*",
                }
                try:
                    extract_data = await asyncio.to_thread(_fetch_json, base_url, extract_params)
                    pages = extract_data.get("query", {}).get("pages", {})
                    page = pages.get(str(page_id), {})
                    extract = page.get("extract", "")
                    if extract:
                        lines.append(f"   Extract: {extract[:300]}")
                except Exception as exc:
                    logger.debug("Failed to fetch extract for %s: %s", title, exc)

            lines.append("")

        return "\n".join(lines)

    async def _execute_read(
        self,
        base_url: str,
        language: str,
        title: Optional[str],
        *,
        read_limit: int,
        offset: Optional[int],
        lines_start: Optional[int],
        lines_end: Optional[int],
        pattern: Optional[str],
        context_lines: Optional[int],
    ) -> str:
        """Fetch the complete plain-text article for ``title``.

        ``prop=extracts&explaintext`` without ``exsentences``/``exintro``
        returns the whole article; ``exsectionformat=wiki`` keeps
        ``== Heading ==`` markers so the agent can jump via ``pattern`` or
        line ranges. Output is paged by ``read_partial_text`` exactly like
        ``rag_search`` document reads.
        """
        title = (title or "").strip()
        if not title:
            return "A title is required for action='read'."

        read_params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": "true",
            "exsectionformat": "wiki",
            "redirects": "1",
            "titles": title,
            "format": "json",
            "origin": "*",
        }
        try:
            data = await asyncio.to_thread(_fetch_json, base_url, read_params)
        except Exception as exc:
            logger.warning("Wikipedia read failed for '%s': %s", title, exc)
            return f"Read failed for '{title}': {exc}"

        pages = data.get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {}) if pages else {}
        resolved_title = page.get("title") or title
        extract = page.get("extract", "")
        if "missing" in page or not extract.strip():
            return (
                f"Wikipedia article not found: '{title}'. "
                "Run action='search' to find the exact title."
            )

        url = f"https://{language}.wikipedia.org/wiki/{urllib.parse.quote(resolved_title.replace(' ', '_'))}"
        limit = min(max(int(read_limit or 0), 1000), _READ_LIMIT_MAX)
        body = read_partial_text(
            extract,
            limit=limit,
            offset=offset,
            lines_start=lines_start,
            lines_end=lines_end,
            line_numbers=True,
            pattern=pattern,
            context_lines=context_lines if context_lines is not None else 2,
            hard_limit=limit,
        )
        return f"Wikipedia article: {resolved_title} ({language})\nURL: {url}\n\n{body}"
