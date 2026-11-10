"""Web scraper builtin tool."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any, Dict

from .common import (
    _CACHE_TTL_SECONDS,
    _url_cache,
    fetch_with_js,
    fetch_with_retry,
)

logger = logging.getLogger(__name__)


class WebScraperTool:
    """Fetch a URL and extract readable text content."""

    name = "web_scraper"
    description = "Fetch the content of a specific URL and return the readable text. Use ONLY 'url' (required) and optional 'render_js' (default false)."

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
                        "url": {
                            "type": "string",
                            "description": "The URL to fetch and read.",
                        },
                        "render_js": {
                            "type": "boolean",
                            "description": "Render JavaScript before extracting text (default false).",
                            "default": False,
                        },
                    },
                    "required": ["url"],
                },
            },
        }

    async def execute(self, url: str, render_js: bool = False) -> str:
        if not url.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")

        cache_key = hashlib.sha256(f"{url}:{render_js}".encode()).hexdigest()
        cached = _url_cache.get(cache_key)
        if cached:
            ts, content = cached
            if (asyncio.get_event_loop().time() - ts) < _CACHE_TTL_SECONDS:
                logger.debug("Cache hit for %s", url)
                return content

        if render_js:
            try:
                text = await fetch_with_js(url)
            except Exception as exc:
                exc_msg = str(exc).split("\n")[0].strip()
                logger.info("JS rendering failed for %s: %s, falling back to static", url, exc_msg)
                text = await fetch_with_retry(url)
        else:
            text = await fetch_with_retry(url)

        _url_cache[cache_key] = (asyncio.get_event_loop().time(), text)
        return text
