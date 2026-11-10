"""RSS feed reader builtin tool."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

import aiohttp
import feedparser

from .common import get_headers, strip_html_tags
from .web_scraper import WebScraperTool

logger = logging.getLogger(__name__)


class RssFeedReaderTool:
    """Fetch and parse an RSS or Atom feed, optionally scraping full article content."""

    name = "rss_feed_reader"
    description = (
        "Fetch and parse an RSS or Atom feed. Returns recent entries with title, "
        "link, published date, and summary. Use ONLY 'url' (required), 'entries' "
        "(number of items, default 10), and 'fetch_full_content' (scrape full article "
        "text, default false)."
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
                        "url": {
                            "type": "string",
                            "description": "The RSS or Atom feed URL.",
                        },
                        "entries": {
                            "type": "integer",
                            "description": "Number of entries to return (default 10).",
                            "default": 10,
                        },
                        "fetch_full_content": {
                            "type": "boolean",
                            "description": "Fetch the full article text for each entry (default false).",
                            "default": False,
                        },
                    },
                    "required": ["url"],
                },
            },
        }

    async def execute(self, url: str, entries: int = 10, fetch_full_content: bool = False) -> str:
        if not url.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")

        headers = get_headers()
        headers["Accept"] = "application/rss+xml,application/atom+xml,application/xml,text/xml,*/*;q=0.9"
        timeout = aiohttp.ClientTimeout(total=60)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"Feed returned HTTP {resp.status}")
                    feed_xml = await resp.text()
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch feed: {exc}")

        parsed = feedparser.parse(feed_xml)
        if parsed.get("bozo") and not parsed.entries:
            exc = parsed.get("bozo_exception")
            raise RuntimeError(f"Failed to parse feed: {exc}")

        feed_entries = parsed.entries[:entries]
        if not feed_entries:
            return f"No entries found in feed: {url}"

        output_lines: List[str] = [
            f"Feed: {parsed.feed.get('title', url)}",
            f"URL: {url}",
            "",
        ]

        scraper = WebScraperTool()
        semaphore = asyncio.Semaphore(3)

        async def _fetch_one(entry: Dict[str, Any], idx: int) -> List[str]:
            lines: List[str] = []
            title = entry.get("title", "Untitled")
            link = entry.get("link", "")
            published = entry.get("published", entry.get("updated", ""))
            summary = entry.get("summary", entry.get("description", ""))

            lines.append(f"{idx}. {title}")
            if link:
                lines.append(f"   Link: {link}")
            if published:
                lines.append(f"   Published: {published}")
            if summary:
                clean_summary = strip_html_tags(summary)
                if clean_summary:
                    lines.append(f"   Summary: {clean_summary[:300]}")

            if fetch_full_content and link:
                async with semaphore:
                    try:
                        page_text = await scraper.execute(url=link)
                        truncated = page_text[:2_000]
                        if len(page_text) > 2_000:
                            truncated += "\n[...truncated]"
                        indented = truncated.replace("\n", "\n   ")
                        lines.append(f"   Full content:\n   {indented}")
                    except Exception as exc:
                        lines.append(f"   [Could not fetch full content: {exc}]")
            lines.append("")
            return lines

        tasks = [_fetch_one(entry, i) for i, entry in enumerate(feed_entries, 1)]
        results = await asyncio.gather(*tasks)
        for lines in results:
            output_lines.extend(lines)

        return "\n".join(output_lines)
