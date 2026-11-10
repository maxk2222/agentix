"""Web search builtin tool."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

from .common import (
    _CACHE_TTL_SECONDS,
    bing_search,
    duckduckgo_search,
    merge_search_results,
)
from .serpapi_search import serpapi_api_key, serpapi_search
from .web_scraper import WebScraperTool

logger = logging.getLogger(__name__)

# In-memory search result cache: key -> (timestamp, (results, engines, errors))
_search_cache: Dict[str, Any] = {}

_SCRAPING_ENGINES = ("duckduckgo", "bing")
_ALL_ENGINES = ("serpapi", "duckduckgo", "bing")
_PAGE_TEXT_LIMIT = 5_000


def _engine_order(preferred: str, serpapi_available: bool) -> List[str]:
    """Engine attempt order: the configured engine first, rest as fallback."""
    if preferred in _ALL_ENGINES:
        order = [preferred] + [e for e in _ALL_ENGINES if e != preferred]
    else:
        order = list(_ALL_ENGINES)
    if not serpapi_available and "serpapi" in order:
        order.remove("serpapi")
    return order


def _truncate_page(text: str, limit: int = _PAGE_TEXT_LIMIT) -> str:
    """Cut extracted page text at a sentence/line boundary instead of mid-word."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    boundary = max(cut.rfind(". "), cut.rfind("\n"))
    if boundary > int(limit * 0.6):
        cut = cut[: boundary + 1]
    return cut.rstrip() + "\n\n... (truncated)"


class WebSearchTool:
    """Search the web via SerpAPI (if configured) and/or DuckDuckGo+Bing.

    Engine selection is configurable through the tool config:

    - ``engine``: ``auto`` (default), ``serpapi``, ``duckduckgo`` or ``bing``.
      ``auto`` prefers SerpAPI when ``SERPAPI_API_KEY`` is set and queries
      DuckDuckGo + Bing in parallel otherwise, merging results with
      deduplication and domain diversity.
    - ``region``: e.g. ``de-de`` — maps to engine-specific locale parameters.
    - ``safe_search``: boolean, default true.
    """

    name = "web_search"
    description = (
        "Search the web for a query. Returns search results with titles, snippets and URLs. "
        "Use ONLY 'query' (required), 'results' (number of links, default 15), and 'read_pages' (pages to fetch, default 0)."
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
                            "description": "The search query. Supports operators like site:example.com.",
                        },
                        "results": {
                            "type": "integer",
                            "description": "Number of result links to return (default 15, max 30).",
                            "default": 15,
                        },
                        "read_pages": {
                            "type": "integer",
                            "description": "How many of the top result pages to fetch and read (default 0).",
                            "default": 0,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    async def _run_search(
        self,
        query: str,
        num: int,
        order: List[str],
        region: str,
        safe_search: bool,
    ) -> Tuple[List[Dict[str, str]], List[str], Dict[str, str]]:
        """Run the engine chain. Returns (merged results, engines used, per-engine errors)."""
        errors: Dict[str, str] = {}
        rest = list(order)

        # SerpAPI first when prioritized — a single high-quality API call.
        if rest and rest[0] == "serpapi":
            rest.pop(0)
            try:
                api_results = await serpapi_search(query, num, region=region)
                if api_results:
                    for item in api_results:
                        item["engine"] = "serpapi"
                    return api_results[:num], ["serpapi"], errors
                errors["serpapi"] = "no results"
            except Exception as exc:
                errors["serpapi"] = str(exc).split("\n")[0][:160]
                logger.debug("SerpAPI search failed: %s", exc)

        scraping = [e for e in rest if e in _SCRAPING_ENGINES]
        if not scraping:
            return [], [], errors

        calls = [
            duckduckgo_search(query, num, region=region, safe_search=safe_search)
            if engine == "duckduckgo"
            else bing_search(query, num, region=region, safe_search=safe_search)
            for engine in scraping
        ]
        outcomes = await asyncio.gather(*calls, return_exceptions=True)

        engine_results: List[Tuple[str, List[Dict[str, str]]]] = []
        succeeded: List[str] = []
        for engine, outcome in zip(scraping, outcomes):
            if isinstance(outcome, Exception):
                errors[engine] = str(outcome).split("\n")[0][:160]
                logger.debug("%s search failed: %s", engine, outcome)
            else:
                succeeded.append(engine)
                engine_results.append((engine, outcome))

        merged = merge_search_results(engine_results, num)
        # An engine that answered but contributed nothing still counts as used.
        used = [e for e, res in engine_results if res]
        return merged, used or succeeded, errors

    async def execute(
        self,
        query: str,
        results: int = 15,
        read_pages: int = 0,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cfg = (context or {}).get("tool_config") or {}
        engine_cfg = str(cfg.get("engine", "auto") or "auto").lower()
        region = str(cfg.get("region", "") or "").strip()
        safe_search = bool(cfg.get("safe_search", True))
        results = max(1, min(30, int(results)))

        order = _engine_order(engine_cfg, bool(serpapi_api_key()))

        cache_key = hashlib.sha256(
            f"{query}|{results}|{engine_cfg}|{region}|{safe_search}".encode("utf-8")
        ).hexdigest()
        cached = _search_cache.get(cache_key)
        if cached and (asyncio.get_event_loop().time() - cached[0]) < _CACHE_TTL_SECONDS:
            merged, engines, errors = cached[1]
        else:
            merged, engines, errors = await self._run_search(
                query, results, order, region, safe_search
            )
            if merged:
                _search_cache[cache_key] = (asyncio.get_event_loop().time(), (merged, engines, errors))

        if not merged:
            if errors:
                detail = "; ".join(f"{e}: {m}" for e, m in errors.items())
                raise RuntimeError(
                    f"Web search unavailable ({detail}). Try again later or rephrase the query."
                )
            return {
                "content": f"Search results for: {query}\n\nNo results found.",
                "metadata": {"engines": engines, "errors": errors},
            }

        output_lines: List[str] = [f"Search results for: {query}", ""]
        for idx, item in enumerate(merged, 1):
            output_lines.append(f"{idx}. {item['title']}")
            output_lines.append(f"   URL: {item['url']}")
            snippet = item.get("snippet", "")
            if snippet:
                output_lines.append(f"   {snippet}")
            output_lines.append("")

        if read_pages > 0:
            scraper = WebScraperTool()
            semaphore = asyncio.Semaphore(3)

            async def _scrape_one(item: Dict[str, str]) -> str:
                url = item["url"]
                if not url.startswith(("http://", "https://")):
                    return f"\nPage: {url or '(unresolved link)'}\n[Skipped: not a fetchable URL]"
                async with semaphore:
                    try:
                        page_text = await scraper.execute(url=url)
                        return f"\nPage: {url}\n{_truncate_page(page_text)}"
                    except Exception as exc:
                        return f"\nPage: {url}\n[Error reading page: {exc}]"

            tasks = [_scrape_one(item) for item in merged[:read_pages]]
            output_lines.append("--- Fetched page content ---")
            output_lines.extend(await asyncio.gather(*tasks))

        if errors:
            detail = "; ".join(f"{e}: {m}" for e, m in errors.items())
            output_lines.append(f"\n[note: some engines failed — {detail}]")

        return {
            "content": "\n".join(output_lines),
            "metadata": {"engines": engines, "errors": errors},
        }
