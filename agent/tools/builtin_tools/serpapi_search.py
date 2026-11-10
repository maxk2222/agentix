"""SerpAPI search builtin tool."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


def serpapi_api_key() -> str:
    """Configured SerpAPI key from the environment (empty when unset)."""
    return os.environ.get("SERPAPI_API_KEY", "")


async def serpapi_search(
    query: str,
    num: int = 5,
    *,
    api_key: str = "",
    region: str = "",
) -> List[Dict[str, str]]:
    """Fetch Google organic results via SerpAPI as title/url/snippet dicts.

    Raises RuntimeError when the request fails. ``region`` like ``"de-de"``
    maps to the ``gl`` (country) and ``hl`` (language) parameters.
    """
    key = api_key or serpapi_api_key()
    if not key:
        raise RuntimeError(
            "SerpAPI key not configured (set 'api_key' in the tool config or SERPAPI_API_KEY)"
        )

    params: Dict[str, Any] = {
        "q": query,
        "api_key": key,
        "engine": "google",
        "num": num,
        "output": "json",
    }
    if region:
        parts = region.split("-")
        params["hl"] = parts[0].lower()
        if len(parts) == 2:
            params["gl"] = parts[1].lower()

    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get("https://serpapi.com/search", params=params) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"SerpAPI returned HTTP {resp.status}: {text[:200]}")
            data = await resp.json()

    results: List[Dict[str, str]] = []
    for item in data.get("organic_results", [])[:num]:
        results.append({
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
        })
    return results


class SerpApiSearchTool:
    """Search via SerpAPI (serpapi.com). Requires an API key."""

    name = "serpapi_search"
    description = "Search the web using SerpAPI. Requires a SerpAPI API key in tool config. Use ONLY 'query' (required) and optional 'num' for number of results."

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
                            "description": "The search query.",
                        },
                        "num": {
                            "type": "integer",
                            "description": "Number of results (default 15).",
                            "default": 15,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    async def execute(
        self,
        query: str,
        num: int = 15,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        cfg_key = str(
            ((context or {}).get("tool_config") or {}).get("api_key") or ""
        ).strip()
        organic = await serpapi_search(query, num, api_key=cfg_key)
        if not organic:
            return f"No results found for: {query}"

        lines = [f"Search results for: {query}", ""]
        for idx, item in enumerate(organic, 1):
            lines.append(f"{idx}. {item['title']}")
            lines.append(f"   URL: {item['url']}")
            lines.append(f"   {item['snippet']}")
            lines.append("")
        return "\n".join(lines)
