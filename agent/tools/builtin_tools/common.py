"""Common helpers for built-in agent tools."""

from __future__ import annotations

import asyncio
import base64
import html as _html_lib
import logging
import random
import re
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

# aiohttp default limit for header line length
_MAX_HEADER_LENGTH = 8190


def _header_too_long(err: Exception) -> bool:
    """Check if an exception is caused by response headers exceeding aiohttp's limit."""
    msg = str(getattr(err, "message", err))
    return "Header value is too long" in msg or "Got more than" in msg


async def _fetch_with_urllib(url: str) -> str:
    """Synchronous fetch fallback using urllib for sites with oversized headers."""
    import urllib.request

    def _sync_fetch() -> str:
        req = urllib.request.Request(url, headers=get_headers())
        with urllib.request.urlopen(req, timeout=60) as resp:
            content_type = resp.headers.get("Content-Type", "")
            html = resp.read().decode("utf-8", errors="replace")
            if "text/html" in content_type:
                return extract_text_from_html(html)
            elif "text/" in content_type:
                return html[:50_000]
            else:
                raise RuntimeError(f"Unsupported content type: {content_type}")

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync_fetch)

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_RETRY_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_BASE = 1.5

# Simple in-memory URL cache: url_hash -> (timestamp, content)
_url_cache: Dict[str, Any] = {}
_CACHE_TTL_SECONDS = 300  # 5 min


def get_headers() -> Dict[str, str]:
    """Build request headers with a random user agent."""
    return {
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
    }


async def fetch_with_retry(
    url: str,
    timeout: aiohttp.ClientTimeout = aiohttp.ClientTimeout(total=60),
) -> str:
    """Fetch a URL with retries, backoff and rotating user agents."""
    last_exc: Optional[Exception] = None

    for attempt in range(1, _RETRY_MAX_ATTEMPTS + 1):
        headers = get_headers()
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as resp:
                    if resp.status == 200:
                        content_type = resp.headers.get("Content-Type", "")
                        if "text/html" in content_type:
                            html = await resp.text()
                            return extract_text_from_html(html)
                        elif "text/" in content_type:
                            text = await resp.text()
                            return text[:50_000]
                        else:
                            raise RuntimeError(f"Unsupported content type: {content_type}")
                    if resp.status in _RETRYABLE_STATUS:
                        last_exc = RuntimeError(f"HTTP {resp.status} for {url}")
                        logger.debug("Retryable HTTP %s for %s (attempt %d/%d)", resp.status, url, attempt, _RETRY_MAX_ATTEMPTS)
                        await asyncio.sleep(_RETRY_BACKOFF_BASE ** attempt)
                        continue
                    # Non-retryable
                    raise RuntimeError(f"HTTP {resp.status} for {url}")
        except (aiohttp.ClientConnectorError, aiohttp.ServerTimeoutError, asyncio.TimeoutError) as exc:
            last_exc = exc
            logger.debug("Network error for %s (attempt %d/%d): %s", url, attempt, _RETRY_MAX_ATTEMPTS, exc)
            await asyncio.sleep(_RETRY_BACKOFF_BASE ** attempt)
            continue
        except aiohttp.ClientResponseError as exc:
            if _header_too_long(exc):
                logger.debug("Header too long for %s (attempt %d/%d), falling back to urllib", url, attempt, _RETRY_MAX_ATTEMPTS)
                try:
                    return await _fetch_with_urllib(url)
                except Exception as fallback_exc:
                    last_exc = fallback_exc
                    logger.debug("urllib fallback failed for %s: %s", url, fallback_exc)
                    await asyncio.sleep(_RETRY_BACKOFF_BASE ** attempt)
                    continue
            raise

    raise last_exc or RuntimeError(f"Failed to fetch {url} after {_RETRY_MAX_ATTEMPTS} attempts")


async def fetch_with_js(url: str) -> str:
    """Fetch a URL using Playwright to render JavaScript.

    Falls back to static fetching if Playwright is not installed.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise RuntimeError("Playwright not installed; install it to use render_js")

    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(headless=True)
        except Exception as exc:
            msg = str(exc).split("\n")[0].strip()
            if "Executable doesn't exist" in msg:
                raise RuntimeError("Playwright browser not installed; run 'playwright install'")
            raise RuntimeError(f"Playwright failed: {msg}")
        try:
            context = await browser.new_context(
                user_agent=random.choice(_USER_AGENTS),
                viewport={"width": 1280, "height": 720},
            )
            page = await context.new_page()
            try:
                await page.goto(url, wait_until="networkidle", timeout=10_000)
                content = await page.content()
                return extract_text_from_html(content)
            except Exception:
                # If networkidle times out, try with less strict wait
                await page.goto(url, wait_until="domcontentloaded", timeout=10_000)
                content = await page.content()
                return extract_text_from_html(content)
        finally:
            await browser.close()


class SearchEngineError(Exception):
    """A search engine request failed, was blocked, or returned unparseable markup."""


# Markers found in search-engine anti-bot / captcha responses instead of results.
_SEARCH_ANOMALY_MARKERS = (
    "anomaly",
    "captcha",
    "challenge-form",
    "b_captcha",
    "are you a robot",
    "verify you are",
)

# Query parameters that add tracking noise to result URLs.
_TRACKING_PARAMS = frozenset({
    "fbclid", "gclid", "dclid", "mc_cid", "mc_eid", "msclkid", "igshid",
    "yclid", "wickedid", "_ga", "_gl", "vero_id", "mkt_tok", "rut",
})

_SEARCH_TIMEOUT = aiohttp.ClientTimeout(total=30)


def _is_anomaly_response(status: int, body: str) -> bool:
    """Detect anti-bot / captcha pages returned instead of real results."""
    if status != 200:
        return True
    probe = body[:8000].lower()
    return any(marker in probe for marker in _SEARCH_ANOMALY_MARKERS)


def _clean_text(raw: str) -> str:
    """Unescape HTML entities, strip tags and collapse whitespace."""
    text = _html_lib.unescape(raw or "")
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_tracking_params(url: str) -> str:
    try:
        parts = urllib.parse.urlsplit(url)
        kept = [
            (k, v)
            for k, v in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
            if k.lower() not in _TRACKING_PARAMS and not k.lower().startswith("utm_")
        ]
        return urllib.parse.urlunsplit(
            (parts.scheme, parts.netloc, parts.path, urllib.parse.urlencode(kept), parts.fragment)
        )
    except Exception:
        return url


def resolve_search_url(href: str) -> str:
    """Resolve a raw search-result href to the real destination URL.

    Handles DuckDuckGo redirect links (``uddg`` param), Bing tracker links
    (``/ck/a`` with base64 ``u`` param), scheme-relative URLs and HTML-escaped
    values. Returns ``""`` when no http(s) destination can be resolved.
    """
    if not href:
        return ""
    href = _html_lib.unescape(href).strip()
    if href.startswith("//"):
        href = "https:" + href
    if not href.startswith(("http://", "https://")):
        return ""
    try:
        parts = urllib.parse.urlsplit(href)
    except Exception:
        return ""
    host = parts.netloc.lower()
    if not host:
        return ""
    query = urllib.parse.parse_qs(parts.query)

    if (host == "duckduckgo.com" or host.endswith(".duckduckgo.com")) and query.get("uddg"):
        target = query["uddg"][0].strip()
        if target.startswith(("http://", "https://")):
            return _strip_tracking_params(target)
        return ""

    if (host == "bing.com" or host.endswith(".bing.com")) and parts.path.startswith("/ck/"):
        encoded = (query.get("u") or [""])[0]
        if encoded.startswith("a1"):
            payload = encoded[2:]
            payload += "=" * (-len(payload) % 4)
            try:
                decoded = base64.b64decode(payload).decode("utf-8", "replace")
            except Exception:
                decoded = ""
            if decoded.startswith(("http://", "https://")):
                return _strip_tracking_params(decoded)
        return ""

    return _strip_tracking_params(href)


def canonical_url(url: str) -> str:
    """Canonical form for result deduplication (host sans www, no fragment)."""
    try:
        parts = urllib.parse.urlsplit(_strip_tracking_params(url))
    except Exception:
        return url
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = parts.path.rstrip("/") or "/"
    return urllib.parse.urlunsplit((parts.scheme.lower(), host, path, parts.query, ""))


def _url_host(url: str) -> str:
    try:
        host = urllib.parse.urlsplit(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def merge_search_results(
    engine_results: List[Tuple[str, List[Dict[str, str]]]],
    max_results: int,
    max_per_domain: int = 2,
) -> List[Dict[str, str]]:
    """Merge result lists round-robin by rank with dedup and domain diversity.

    ``engine_results`` is an ordered list of ``(engine_name, results)``. The
    per-domain cap is applied first, then relaxed to still fill ``max_results``.
    Each result gets an ``engine`` marker with its origin.
    """
    merged: List[Dict[str, str]] = []
    seen: set = set()
    domain_count: Dict[str, int] = {}
    longest = max((len(results) for _, results in engine_results), default=0)

    for domain_cap in (max_per_domain, None):
        for rank in range(longest):
            for engine, results in engine_results:
                if rank >= len(results):
                    continue
                item = results[rank]
                url = item.get("url") or ""
                key = canonical_url(url)
                if key in seen:
                    continue
                host = _url_host(key)
                if domain_cap is not None and domain_count.get(host, 0) >= domain_cap:
                    continue
                seen.add(key)
                domain_count[host] = domain_count.get(host, 0) + 1
                merged.append({**item, "engine": engine})
                if len(merged) >= max_results:
                    return merged
    return merged


_ANCHOR_RE = re.compile(r"<a\b([^>]*)>(.*?)</a>", re.DOTALL | re.IGNORECASE)
_ATTR_RE = re.compile(r'([\w-]+)\s*=\s*"([^"]*)"')


def _anchor_attrs(attr_src: str) -> Dict[str, str]:
    return {k.lower(): v for k, v in _ATTR_RE.findall(attr_src)}


def _parse_ddg_markup(
    html: str,
    link_marker: str,
    snippet_marker: str,
    max_results: int,
) -> List[Dict[str, str]]:
    """Parse DuckDuckGo result anchors regardless of attribute order.

    ``link_marker``/``snippet_marker`` are class tokens (``result__a`` /
    ``result__snippet`` for the html endpoint, ``result-link`` /
    ``result-snippet`` for the lite endpoint). Snippets are located in the
    segment between a result link and the next one, which tolerates results
    without snippets without shifting the pairing.
    """
    anchors: List[Tuple[int, int, str, str]] = []
    for m in _ANCHOR_RE.finditer(html):
        attrs = _anchor_attrs(m.group(1))
        if link_marker in attrs.get("class", ""):
            anchors.append((m.start(), m.end(), attrs.get("href", ""), m.group(2)))

    snippet_re = re.compile(
        r'class="[^"]*' + re.escape(snippet_marker) + r'[^"]*"[^>]*>(.*?)</(?:a|td|div|span)\s*>',
        re.DOTALL | re.IGNORECASE,
    )
    results: List[Dict[str, str]] = []
    seen: set = set()
    for idx, (_start, end, href, title_html) in enumerate(anchors):
        if len(results) >= max_results:
            break
        url = resolve_search_url(href)
        if not url:
            continue
        canon = canonical_url(url)
        if canon in seen:
            continue
        seen.add(canon)

        boundary = anchors[idx + 1][0] if idx + 1 < len(anchors) else len(html)
        snippet = ""
        snippet_match = snippet_re.search(html[end:boundary])
        if snippet_match:
            snippet = _clean_text(snippet_match.group(1))

        title = _clean_text(title_html) or _url_host(url)
        results.append({"title": title, "url": url, "snippet": snippet})
    return results


async def duckduckgo_search(
    query: str,
    num_results: int = 5,
    *,
    region: str = "",
    safe_search: bool = True,
) -> List[Dict[str, str]]:
    """Search via DuckDuckGo (html endpoint, lite fallback).

    Raises :class:`SearchEngineError` when every endpoint fails or is blocked;
    returns a (possibly empty) result list on a healthy response.
    """
    data: Dict[str, str] = {"q": query, "kp": "-1" if safe_search else "-2"}
    if region:
        data["kl"] = region.lower()

    errors: List[str] = []
    for endpoint, kind in (
        ("https://html.duckduckgo.com/html/", "html"),
        ("https://lite.duckduckgo.com/lite/", "lite"),
    ):
        for attempt in range(2):
            try:
                async with aiohttp.ClientSession(timeout=_SEARCH_TIMEOUT) as session:
                    async with session.post(endpoint, data=data, headers=get_headers()) as resp:
                        status, body = resp.status, await resp.text()
            except Exception as exc:
                errors.append(f"{kind}: {exc}")
                await asyncio.sleep(0.8 * (attempt + 1))
                continue
            if _is_anomaly_response(status, body):
                errors.append(f"{kind}: blocked (HTTP {status}, anomaly page)")
                await asyncio.sleep(0.8 * (attempt + 1))
                continue
            parser = parse_duckduckgo_results if kind == "html" else parse_duckduckgo_lite_results
            return parser(body, max_results=num_results)

    raise SearchEngineError("DuckDuckGo failed: " + "; ".join(dict.fromkeys(errors)))


def parse_duckduckgo_results(html: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Extract results from the DuckDuckGo ``html`` endpoint markup."""
    return _parse_ddg_markup(html, "result__a", "result__snippet", max_results)


def parse_duckduckgo_lite_results(html: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Extract results from the DuckDuckGo ``lite`` endpoint markup."""
    return _parse_ddg_markup(html, "result-link", "result-snippet", max_results)


async def bing_search(
    query: str,
    num_results: int = 5,
    *,
    region: str = "",
    safe_search: bool = True,
) -> List[Dict[str, str]]:
    """Search via Bing HTML. Raises :class:`SearchEngineError` on failure."""
    params: Dict[str, str] = {
        "q": query,
        # Over-fetch to compensate for ad segments filtered out by the parser.
        "count": str(num_results + 8),
        "safeSearch": "Moderate" if safe_search else "Off",
    }
    if region:
        parts = region.split("-")
        if len(parts) == 2:
            params["mkt"] = f"{parts[0].lower()}-{parts[1].upper()}"
            params["setlang"] = parts[0].lower()

    errors: List[str] = []
    for attempt in range(2):
        try:
            async with aiohttp.ClientSession(timeout=_SEARCH_TIMEOUT) as session:
                async with session.get(
                    "https://www.bing.com/search", params=params, headers=get_headers()
                ) as resp:
                    status, body = resp.status, await resp.text()
        except Exception as exc:
            errors.append(str(exc))
            await asyncio.sleep(0.8 * (attempt + 1))
            continue
        if _is_anomaly_response(status, body):
            errors.append(f"blocked (HTTP {status}, anomaly page)")
            await asyncio.sleep(0.8 * (attempt + 1))
            continue
        return parse_bing_results(body, max_results=num_results)

    raise SearchEngineError("Bing failed: " + "; ".join(dict.fromkeys(errors)))


def parse_bing_results(html: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Extract title, URL and snippet from Bing HTML."""
    results: List[Dict[str, str]] = []
    seen_urls: set = set()

    # Bing results are typically in <li class="b_algo"> blocks
    # Each block contains: <h2><a href="...">Title</a></h2> and <p>Snippet</p>
    # Sponsored entries keep the b_algo wrapper but carry ad markers inside —
    # strip ad-flagged segments (and dedicated b_ad blocks) up front so the
    # loose fallback cannot resurrect them either.
    segment_pattern = re.compile(
        r'<li[^>]*?class="b_algo"[^>]*>(.*?)</li>',
        re.DOTALL | re.IGNORECASE,
    )
    ad_markers = re.compile(r'b_adlabel|b_adSlug|b_adTop|b_adBottom', re.IGNORECASE)

    def _drop_ads(match: "re.Match") -> str:
        return "" if ad_markers.search(match.group(1)) else match.group(0)

    cleaned = segment_pattern.sub(_drop_ads, html)
    cleaned = re.sub(
        r'<li[^>]*?class="b_ad[^"]*"[^>]*>.*?</li>',
        "", cleaned, flags=re.DOTALL | re.IGNORECASE,
    )
    inner_pattern = re.compile(
        r'<h2[^>]*>.*?<a[^>]*?href="([^"]+)"[^>]*>(.*?)</a>.*?</h2>.*?<p[^>]*>(.*?)</p>',
        re.DOTALL | re.IGNORECASE,
    )

    for segment in segment_pattern.finditer(cleaned):
        if len(results) >= max_results:
            break
        match = inner_pattern.search(segment.group(1))
        if not match:
            continue
        url = resolve_search_url(match.group(1))
        if not url:
            continue
        canon = canonical_url(url)
        if canon in seen_urls:
            continue

        seen_urls.add(canon)
        results.append({
            "title": _clean_text(match.group(2) or ""),
            "url": url,
            "snippet": _clean_text(match.group(3) or ""),
        })

    if not results:
        # Fallback: looser pattern without <li> wrapper
        loose_pattern = re.compile(
            r'<h2[^>]*>.*?<a[^>]*?href="([^"]+)"[^>]*>(.*?)</a>.*?</h2>',
            re.DOTALL | re.IGNORECASE,
        )
        for match in loose_pattern.finditer(cleaned):
            if len(results) >= max_results:
                break
            url = resolve_search_url(match.group(1))
            if not url:
                continue
            canon = canonical_url(url)
            if canon in seen_urls:
                continue
            seen_urls.add(canon)
            results.append({
                "title": _clean_text(match.group(2) or ""),
                "url": url,
                "snippet": "",
            })

    if not results:
        logger.debug("Bing HTML snippet (first 500 chars): %s", html[:500])

    return results


def extract_text_from_html(html: str) -> str:
    """Structure-aware HTML-to-text extraction.

    Removes boilerplate (nav, footer, header, aside) and prioritises
    article/main/paragraph content.
    """
    # 1. Remove script, style, nav, footer, header, aside blocks
    for tag in ("script", "style", "nav", "footer", "header", "aside"):
        html = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", " ", html, flags=re.DOTALL | re.IGNORECASE)

    # 2. Try to extract article/main first, fall back to whole body
    article_match = re.search(r"<article[^>]*>(.*?)</article>", html, re.DOTALL | re.IGNORECASE)
    main_match = re.search(r"<main[^>]*>(.*?)</main>", html, re.DOTALL | re.IGNORECASE)
    body_match = re.search(r"<body[^>]*>(.*?)</body>", html, re.DOTALL | re.IGNORECASE)

    source = article_match.group(1) if article_match else (main_match.group(1) if main_match else (body_match.group(1) if body_match else html))

    # 3. Extract text from high-value tags with paragraph breaks
    chunks: List[str] = []
    tag_weights = {"h1": 3, "h2": 3, "h3": 2, "p": 1, "li": 1}
    for tag, weight in tag_weights.items():
        for m in re.finditer(rf"<{tag}[^>]*>(.*?)</{tag}>", source, re.DOTALL | re.IGNORECASE):
            text = strip_html_tags(m.group(1))
            if text:
                chunks.append(text + ("\n" * weight))

    # If no high-value tags, fall back to stripping all tags
    if not chunks:
        text = re.sub(r"<[^>]+>", " ", source)
        text = text.replace("&nbsp;", " ").replace("&amp;", "&")
        text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
        text = re.sub(r"\s+", " ", text).strip()
        return text[:50_000]

    result = re.sub(r"\n\s*\n\s*\n+", "\n\n", "\n".join(chunks))
    result = result.strip()
    return result[:50_000]


def strip_html_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", _html_lib.unescape(text)).strip()


def _remaining_read_plan(lines: List[str], next_idx: int, effective_limit: int) -> str:
    """Deterministic paging plan for the rest of the document.

    Greedy maximal line windows under ``effective_limit - reserve`` starting at
    ``next_idx`` (0-based).  Returns e.g. "lines 45-98, 99-152" so the caller
    can issue the exact remaining ``lines_start``/``lines_end`` reads instead
    of improvising small windows.
    """
    budget = max(256, effective_limit - 200)
    windows: List[tuple[int, int]] = []
    i = next_idx
    total = len(lines)
    while i < total and len(windows) < 9:
        start = i
        size = 0
        while i < total and (size + len(lines[i]) + 1 <= budget or i == start):
            size += len(lines[i]) + 1
            i += 1
        windows.append((start + 1, i))  # 1-based inclusive
    if not windows:
        return ""
    plan = ", ".join(
        f"lines {a}-{b}" if a != b else f"line {a}" for a, b in windows[:8]
    )
    if len(windows) > 8:
        plan += f", +{len(windows) - 8} more windows"
    elif i < total:
        plan += f", +{total - i} more lines"
    return plan


def _numbered_lines(lines: List[str], start_line: int) -> str:
    """Prefix every line with its real 1-based number (``NNNN│ ``)."""
    return "\n".join(
        f"{start_line + i:>4}│ {line}" for i, line in enumerate(lines)
    )


def _merge_hit_windows(hits: List[int], ctx: int, total_lines: int) -> List[List[int]]:
    """Merge hit indices into clamped [start, end] context windows."""
    windows: List[List[int]] = []
    for hit in hits:
        s, e = hit - ctx, hit + ctx
        if windows and s <= windows[-1][1] + 1:
            windows[-1][1] = max(windows[-1][1], e)
        else:
            windows.append([s, e])
    return [[max(0, s), min(total_lines - 1, e)] for s, e in windows]


def _pattern_excerpt(
    text: str,
    pattern: Optional[str] = None,
    *,
    patterns: Optional[List[str]] = None,
    context_lines: int = 2,
    hard_limit: int = 15_000,
) -> str:
    """Grep-style excerpt: matching lines with context, numbered, merged.

    Case-insensitive substring match. Overlapping context windows merge;
    windows are rendered until ``hard_limit`` is reached (the first window is
    always included even if it alone exceeds the limit). ``patterns`` renders
    one labeled group per pattern so a single call can verify many needles —
    unmatched patterns are listed explicitly.
    """
    needles = [p for p in ([pattern] if pattern else []) + (patterns or []) if p]
    if not needles:
        return "(no pattern given)"
    if len(needles) == 1:
        return _single_pattern_excerpt(
            text, needles[0], context_lines=context_lines, hard_limit=hard_limit
        )
    lines = text.splitlines()
    total_lines = len(lines)
    ctx = max(0, min(int(context_lines or 0), 10))
    rendered: List[str] = []
    unchecked: List[str] = []
    for i, pat in enumerate(needles):
        needle = pat.casefold()
        hits = [j for j, ln in enumerate(lines) if needle in ln.casefold()]
        if not hits:
            block = f"── '{pat}' — no matches ──"
        else:
            parts = [f"── '{pat}' ({len(hits)} matches) ──"]
            for s, e in _merge_hit_windows(hits, ctx, total_lines):
                parts.append(_numbered_lines(lines[s:e + 1], s + 1))
            block = "\n   ⋮\n".join(parts)
        candidate = block if not rendered else "\n\n".join(rendered + [block])
        if rendered and len(candidate) > hard_limit:
            unchecked = needles[i:]
            break
        rendered.append(block)
        if len(candidate) > hard_limit:
            unchecked = needles[i + 1:]
            break
    body = "\n\n".join(rendered)
    if len(body) > hard_limit:
        nl = body.rfind("\n", 0, hard_limit)
        body = body[:nl] if nl > 0 else body[:hard_limit]
    if unchecked:
        body += (
            f"\n\n… (checked {len(needles) - len(unchecked)} of {len(needles)} "
            f"patterns; unchecked: {', '.join(unchecked)} — narrow patterns or raise limit)"
        )
    else:
        body += f"\n\n… (checked {len(needles)} patterns across {total_lines} lines)"
    return body


def _single_pattern_excerpt(
    text: str,
    pattern: str,
    *,
    context_lines: int = 2,
    hard_limit: int = 15_000,
) -> str:
    lines = text.splitlines()
    total_lines = len(lines)
    ctx = max(0, min(int(context_lines or 0), 10))
    needle = pattern.casefold()
    hits = [i for i, ln in enumerate(lines) if needle in ln.casefold()]
    if not hits:
        return f"(no matches for '{pattern}' in {total_lines} lines)"

    windows = _merge_hit_windows(hits, ctx, total_lines)

    rendered: List[str] = []
    for s, e in windows:
        block = _numbered_lines(lines[s:e + 1], s + 1)
        candidate = block if not rendered else "\n   ⋮\n".join(rendered + [block])
        if rendered and len(candidate) > hard_limit:
            break
        rendered.append(block)
        if len(candidate) > hard_limit:
            break

    body = "\n   ⋮\n".join(rendered)
    if len(body) > hard_limit:
        # a single huge merged window must still respect the cap
        nl = body.rfind("\n", 0, hard_limit)
        body = body[:nl] if nl > 0 else body[:hard_limit]
        return body + (
            f"\n\n… (pattern output truncated at {hard_limit} chars — "
            f"{len(hits)} matches in {len(windows)} window(s); narrow the "
            f"pattern or use lines_start/lines_end)"
        )
    shown_ranges = ", ".join(
        f"{s + 1}-{e + 1}" if s != e else str(s + 1) for s, e in windows[: len(rendered)]
    )
    if len(rendered) < len(windows):
        body += (
            f"\n\n… ({len(hits)} matches in {len(windows)} windows — showing lines "
            f"{shown_ranges}; {len(windows) - len(rendered)} more window(s) cut off — "
            f"narrow the pattern or use lines_start/lines_end)"
        )
    else:
        body += (
            f"\n\n… ({len(hits)} matches in {len(windows)} window(s) — lines "
            f"{shown_ranges} of {total_lines} lines)"
        )
    return body


def read_partial_text(
    text: str,
    *,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    lines_start: Optional[int] = None,
    lines_end: Optional[int] = None,
    hard_limit: int = 15_000,
    line_numbers: bool = False,
    snap_to_lines: bool = True,
    pattern: Optional[str] = None,
    patterns: Optional[List[str]] = None,
    context_lines: int = 2,
) -> str:
    """Slice ``text`` by line range or offset/limit with a navigation footer.

    ``pattern`` (case-insensitive substring) short-circuits everything and
    returns grep-style numbered match windows. ``lines_start``/``lines_end``
    (1-based, inclusive) take priority over ``offset``/``limit``; a negative
    ``offset`` counts from the end. With ``snap_to_lines`` char-based cuts land
    on line boundaries so pages never split words mid-token. Output never
    exceeds ``hard_limit`` characters. Partial results carry a footer with the
    *actual* range returned plus the next ``offset``/``lines_start`` to use;
    reads that reach EOF are labelled "excerpt — end of document".
    ``line_numbers`` prefixes every content line ``NNNN│ `` with its real
    1-based number.
    """
    total_chars = len(text)
    effective_limit = max(1, min(limit or hard_limit, hard_limit))

    if pattern or patterns:
        return _pattern_excerpt(
            text, pattern, patterns=patterns,
            context_lines=context_lines, hard_limit=effective_limit,
        )

    total_lines = text.count("\n") + 1

    if lines_start is not None or lines_end is not None:
        lines = text.splitlines()
        line_count = len(lines)
        start_idx = (lines_start - 1) if lines_start is not None else 0
        end_idx = lines_end if lines_end is not None else line_count
        start_idx = max(0, start_idx)
        end_idx = min(end_idx, line_count)
        if start_idx >= end_idx:
            req_start = lines_start if lines_start is not None else 1
            req_end = lines_end if lines_end is not None else line_count
            return (
                f"(no content: requested lines {req_start}-{req_end} are outside "
                f"the document's {line_count} lines)"
            )
        sel_lines = lines[start_idx:end_idx]
        selected = "\n".join(sel_lines)
        truncated = len(selected) > effective_limit
        shown_end = end_idx
        mid_line = False
        if truncated:
            cut = selected[:effective_limit]
            mid_line = True
            if snap_to_lines:
                nl = cut.rfind("\n")
                if nl > 0:
                    cut = cut[:nl]
                    mid_line = False
            shown_end = start_idx + cut.count("\n") + 1
            sel_lines = [cut] if mid_line else lines[start_idx:shown_end]
            selected = cut
        if start_idx > 0 or end_idx < line_count or truncated:
            marker = "truncated" if truncated else "excerpt"
            if mid_line:
                char_at_cut = sum(len(l) + 1 for l in lines[:start_idx]) + len(cut)
                hint = f" — continue with offset={char_at_cut} (cut inside line {shown_end})"
            elif shown_end < line_count:
                next_char = sum(len(l) + 1 for l in lines[:shown_end])
                hint = f" — continue with lines_start={shown_end + 1} or offset={next_char}"
                plan = _remaining_read_plan(lines, shown_end, effective_limit)
                if plan:
                    hint += f" — read plan: {plan}"
            else:
                hint = " — end of document"
            body = (
                _numbered_lines(sel_lines, start_idx + 1)
                if line_numbers
                else selected
            )
            body += (
                f"\n\n… ({marker} — showing lines {start_idx + 1}-{shown_end} "
                f"of {line_count} lines, {total_chars} chars total{hint})"
            )
            return body
        return _numbered_lines(sel_lines, start_idx + 1) if line_numbers else selected

    effective_offset = offset or 0
    if effective_offset < 0:
        effective_offset = max(0, total_chars + effective_offset)
    if effective_offset >= total_chars:
        return (
            f"(no content: offset {effective_offset} is beyond the document's "
            f"{total_chars} chars)"
        )

    start = effective_offset
    cut_inside_line = False
    if snap_to_lines and start > 0:
        # snap BACK to the start of the line containing the offset so the
        # returned page always covers the line it lands in — forward-snapping
        # would silently skip content whenever callers guess mid-line offsets.
        nl = text.rfind("\n", 0, start)
        if nl != -1:
            start = nl + 1
        else:
            first_nl = text.find("\n")
            if first_nl == -1 or first_nl >= effective_limit:
                # offset lands inside a first line that alone spans the page —
                # keep the raw cut so continuation offsets still make progress
                cut_inside_line = True
            else:
                start = 0  # offset lands inside the first line — show it whole
    if start >= total_chars:
        return (
            f"(no content: offset {effective_offset} is beyond the document's "
            f"{total_chars} chars)"
        )

    raw_end = min(total_chars, start + effective_limit)
    end = raw_end
    if snap_to_lines and raw_end < total_chars:
        nl = text.rfind("\n", start, raw_end)
        if nl > start:
            end = nl
        else:
            cut_inside_line = True
    result = text[start:end]

    start_line = text.count("\n", 0, start) + 1
    end_line = start_line + result.count("\n")
    # EOF is also reached when only whitespace remains after `end`
    reached_eof = end >= total_chars or not text[end:].strip()

    if line_numbers:
        result = _numbered_lines(result.split("\n"), start_line)

    if start == 0 and reached_eof:
        return result  # full document — no footer needed

    label = "excerpt" if reached_eof else "truncated"
    # `end` lands on a newline when snapping succeeded → next offset is end+1;
    # mid-line cuts continue exactly at `end` (the first unshown char).
    next_offset = end + 1 if end < total_chars and text[end] == "\n" else end
    if reached_eof:
        hint = " — end of document"
    else:
        hint = f" — continue with offset={next_offset} or lines_start={end_line + 1}"
        plan = _remaining_read_plan(text.splitlines(), end_line, effective_limit)
        if plan:
            hint += f" — read plan: {plan}"
    note = "; cut inside long line" if cut_inside_line else ""
    return (
        result
        + f"\n\n… ({label} — showing lines {start_line}-{end_line} of {total_lines} · "
        + f"chars {start}-{end} of {total_chars}{note}{hint})"
    )


def normalize_collection_names(value: Any) -> List[str]:
    """Normalise a collection_names config value into a list of strings.

    Handles:
    - list: filters out empty entries and converts to str
    - str: tries JSON-parse first (e.g. '["col_a", "col_b"]'), then comma-split
    - everything else: returns empty list
    """
    if isinstance(value, list):
        return [str(n).strip() for n in value if n and str(n).strip()]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        import json as _json
        try:
            parsed = _json.loads(s)
            if isinstance(parsed, list):
                return [str(n).strip() for n in parsed if n and str(n).strip()]
        except (ValueError, TypeError):
            pass
        return [n.strip() for n in s.split(",") if n.strip()]
    return []


# ---------------------------------------------------------------------------
# Shared multi-edit ("edits" batch) support
# ---------------------------------------------------------------------------
#
# collection_manager and agent_workspace share the same ordered-batch
# semantics: each op transforms the text left by its predecessors and the
# caller persists the result exactly once (all-or-nothing). Nested op fields
# need explicit coercion — validate_and_normalize_args only normalizes
# top-level arguments, so models' stringified ints reach us verbatim.

_MAX_EDITS = 50

_INT_EDIT_FIELDS = ("line", "lines_start", "lines_end", "index", "from", "to")
_STR_EDIT_FIELDS = ("content", "old_text", "text_snippet", "after_text")


def validate_edits_list(edits: Any) -> List[Dict[str, Any]]:
    """Validate an ``edits`` batch: non-empty list of dicts, at most 50."""
    if not isinstance(edits, list):
        raise ValueError("'edits' must be a list of edit objects")
    if not edits:
        raise ValueError("'edits' must be a non-empty list of edit objects")
    if len(edits) > _MAX_EDITS:
        raise ValueError(
            f"too many edits in one call ({len(edits)}, max {_MAX_EDITS}) "
            "— split into multiple calls"
        )
    for i, op in enumerate(edits):
        if not isinstance(op, dict):
            raise ValueError(
                f"edit #{i} must be an object (got {type(op).__name__})"
            )
    return edits


def coerce_edit_args(op: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce nested edit-op fields (stringified ints, non-str text fields)."""
    fixed = dict(op)
    for key in _INT_EDIT_FIELDS:
        val = fixed.get(key)
        if val is None:
            continue
        if isinstance(val, bool):
            raise ValueError(f"'{key}' must be an integer")
        if isinstance(val, int):
            continue
        if isinstance(val, float) and val.is_integer():
            fixed[key] = int(val)
        elif isinstance(val, str):
            try:
                fixed[key] = int(val.strip())
            except ValueError:
                raise ValueError(f"'{key}' must be an integer (got {val!r})")
        else:
            raise ValueError(
                f"'{key}' must be an integer (got {type(val).__name__})"
            )
    for key in _STR_EDIT_FIELDS:
        val = fixed.get(key)
        if val is not None and not isinstance(val, str):
            fixed[key] = str(val)
    return fixed


def edit_insert(
    text: str,
    content: str,
    line: Optional[int] = None,
    after_text: Optional[str] = None,
) -> str:
    if line is not None:
        if not isinstance(line, int) or line == 0 or line < -1:
            raise ValueError(
                "'line' must be >= 1 (insert before that line) or -1 to append at end"
            )
        lines = text.splitlines(keepends=True)
        if line == -1 or line > len(lines):
            return text + content
        insert_idx = line - 1
        insert_idx = max(0, min(insert_idx, len(lines)))
        return "".join(lines[:insert_idx]) + content + "".join(lines[insert_idx:])
    if after_text is not None:
        if not after_text:
            raise ValueError("'after_text' must be a non-empty string")
        idx = text.find(after_text)
        if idx == -1:
            raise ValueError(f"Snippet not found: {after_text!r}. Use 'line' instead.")
        insert_pos = idx + len(after_text)
        return text[:insert_pos] + content + text[insert_pos:]
    raise ValueError("For insert, specify either 'line' or 'after_text'.")


def edit_remove(
    text: str,
    lines_start: Optional[int] = None,
    lines_end: Optional[int] = None,
    text_snippet: Optional[str] = None,
) -> str:
    if lines_start is not None or lines_end is not None:
        if (lines_start is not None and lines_start < 1) or (
            lines_end is not None and lines_end < 1
        ):
            raise ValueError("'lines_start'/'lines_end' must be >= 1")
        if (
            lines_start is not None
            and lines_end is not None
            and lines_end < lines_start
        ):
            raise ValueError(
                f"'lines_end' ({lines_end}) must be >= 'lines_start' ({lines_start})"
            )
        lines = text.splitlines(keepends=True)
        start_idx = (lines_start - 1) if lines_start is not None else 0
        end_idx = lines_end if lines_end is not None else len(lines)
        start_idx = max(0, start_idx)
        end_idx = min(end_idx, len(lines))
        return "".join(lines[:start_idx]) + "".join(lines[end_idx:])
    if text_snippet is not None:
        if not text_snippet:
            raise ValueError("'text_snippet' must be a non-empty string")
        idx = text.find(text_snippet)
        if idx == -1:
            raise ValueError(
                f"Snippet not found: {text_snippet!r}. "
                "Use 'lines_start'/'lines_end' instead."
            )
        return text[:idx] + text[idx + len(text_snippet):]
    raise ValueError("For remove, specify either 'lines_start'/'lines_end' or 'text_snippet'.")


def edit_replace(text: str, old_text: str, content: str) -> str:
    if not old_text:
        raise ValueError("'old_text' must be a non-empty string")
    if old_text not in text:
        raise ValueError(f"Text to replace not found: {old_text!r}")
    return text.replace(old_text, content, 1)


def apply_text_edits(
    text: str, edits: List[Dict[str, Any]]
) -> Tuple[str, List[str]]:
    """Apply ``edits`` sequentially to ``text``; each op sees prior results.

    Returns ``(new_text, op_names)``. Any op failure raises a ValueError
    tagged with the op index and "no changes saved" — callers must not
    persist anything in that case (all-or-nothing batch semantics).
    """
    validate_edits_list(edits)
    current = text
    op_names: List[str] = []
    for i, raw_op in enumerate(edits):
        op_label = "?"
        try:
            op = coerce_edit_args(raw_op)
            op_label = str(op.get("operation") or "").lower().strip() or "(missing)"
            if op_label == "insert":
                content = op.get("content")
                if content is None:
                    raise ValueError("content is required for insert")
                current = edit_insert(
                    current, content,
                    line=op.get("line"), after_text=op.get("after_text"),
                )
            elif op_label == "remove":
                current = edit_remove(
                    current,
                    lines_start=op.get("lines_start"),
                    lines_end=op.get("lines_end"),
                    text_snippet=op.get("text_snippet"),
                )
            elif op_label == "replace":
                content = op.get("content")
                if content is None:
                    raise ValueError("content is required for replace")
                old_text = op.get("old_text")
                if old_text is None:
                    raise ValueError("old_text is required for replace")
                current = edit_replace(current, old_text, content)
            else:
                raise ValueError(
                    f"unknown operation '{op_label}' — use insert|remove|replace"
                )
            op_names.append(op_label)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"edit #{i} ({op_label}) failed: {exc} — no changes saved."
            ) from exc
    return current, op_names
