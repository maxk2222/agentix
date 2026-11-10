"""Tests for agent.tools registry and built-in tools."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch
import sys

import pytest

from agent.tools import ToolRegistry
from agent.tools.builtin import AgentMemoryTool, CollectionManagerTool, WebScraperTool, WebSearchTool, OpenAlexSearchTool, RssFeedReaderTool, SSHTool


class TestToolRegistry:
    def test_builtins_registered(self):
        registry = ToolRegistry()
        builtins = registry.list_tools(enabled_only=True)
        assert "web_scraper" in builtins
        assert "web_search" in builtins
        assert "serpapi_search" in builtins
        assert "openalex_search" in builtins
        assert "rss_feed_reader" in builtins
        assert "code_execution" in builtins
        assert "agent_memory" in builtins
        assert "webhook_post" in builtins
        assert "email_sender" in builtins
        assert "long_document_summarizer" in builtins
        assert "plan_generator" in builtins
        assert "arxiv_search" in builtins
        assert "wikipedia_search" in builtins
        assert "weather_lookup" in builtins
        assert "ssh_exec" in builtins

    def test_get_tool_builtin(self):
        registry = ToolRegistry()
        tool = registry.get_tool("web_scraper")
        assert tool is not None
        assert hasattr(tool, "execute")

    def test_get_tool_missing(self):
        registry = ToolRegistry()
        assert registry.get_tool("nonexistent") is None

    def test_tool_schemas(self):
        registry = ToolRegistry()
        schemas = registry.tool_schemas(["web_scraper", "web_search"])
        assert len(schemas) == 2
        for s in schemas:
            assert "function" in s

    def test_list_tools_includes_custom(self):
        registry = ToolRegistry()
        registry.register_custom("my_tool", MagicMock(enabled=True, tool_schema={}))
        tools = registry.list_tools(enabled_only=True)
        assert "my_tool" in tools

    @pytest.mark.asyncio
    async def test_execute_builtin_mocked(self):
        registry = ToolRegistry()
        mock_tool = MagicMock()
        mock_tool.execute = AsyncMock(return_value="mocked result")
        registry.register_builtin("mock_tool", mock_tool)

        result = await registry.execute("mock_tool", {"url": "http://example.com"})
        assert result["ok"] is True
        assert result["content"] == "mocked result"

    @pytest.mark.asyncio
    async def test_execute_not_found(self):
        registry = ToolRegistry()
        result = await registry.execute("missing", {})
        assert result["ok"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_load_custom_tools(self):
        registry = ToolRegistry()
        mock_store = MagicMock()
        mock_entry = MagicMock()
        mock_entry.kind = "custom"
        mock_entry.name = "custom_tool"
        mock_entry.enabled = True
        mock_entry.tool_schema = {}
        mock_entry.venv_config = None
        mock_entry.runtime_mode = "venv"
        mock_entry.code = "print('hi')"
        mock_store.list_tools = AsyncMock(return_value=[mock_entry])

        await registry.load_custom_tools(mock_store)
        assert registry.get_tool("custom_tool") is not None

    @pytest.mark.asyncio
    async def test_refresh_schemas_calls_tool_refresh(self):
        registry = ToolRegistry()
        mock_tool = MagicMock()
        mock_tool.refresh_schema = AsyncMock()
        registry.register_builtin("refreshable_tool", mock_tool)

        await registry.refresh_schemas()
        mock_tool.refresh_schema.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_refresh_schemas_skips_tools_without_refresh(self):
        registry = ToolRegistry()
        mock_tool = MagicMock()
        del mock_tool.refresh_schema
        registry.register_builtin("no_refresh_tool", mock_tool)

        await registry.refresh_schemas()
        # Should complete without error


class TestWebScraperTool:
    @pytest.mark.asyncio
    async def test_tool_schema(self):
        tool = WebScraperTool()
        ts = tool.tool_schema
        assert ts["type"] == "function"
        assert ts["function"]["name"] == "web_scraper"

    @pytest.mark.asyncio
    async def test_execute_mocked(self):
        tool = WebScraperTool()
        with patch.object(tool, "execute", return_value="Hello World"):
            result = await tool.execute("http://example.com")
        assert "Hello World" in result


class TestWebSearchTool:
    @pytest.fixture(autouse=True)
    def _no_serpapi_and_clean_cache(self):
        """Isolate tests from a real SERPAPI_API_KEY and stale cache entries."""
        from agent.tools.builtin_tools.web_search import _search_cache
        _search_cache.clear()
        with patch("agent.tools.builtin_tools.web_search.serpapi_api_key", return_value=""):
            yield
        _search_cache.clear()

    @pytest.mark.asyncio
    async def test_tool_schema(self):
        tool = WebSearchTool()
        ts = tool.tool_schema
        assert ts["type"] == "function"
        assert ts["function"]["name"] == "web_search"

    @pytest.mark.asyncio
    async def test_execute_mocked(self):
        tool = WebSearchTool()
        with patch.object(tool, "execute", return_value="Example\nexample.com\nA description"):
            result = await tool.execute("test query")
        assert "Example" in result
        assert "example.com" in result

    @pytest.mark.asyncio
    async def test_read_pages_parallel(self):
        tool = WebSearchTool()
        with patch("agent.tools.builtin_tools.web_search.duckduckgo_search", return_value=[
            {"title": "Page 1", "url": "http://a.com", "snippet": "snippet 1"},
            {"title": "Page 2", "url": "http://b.com", "snippet": "snippet 2"},
        ]), patch("agent.tools.builtin_tools.web_search.bing_search", return_value=[]):
            with patch.object(WebScraperTool, "execute", return_value="page text"):
                result = await tool.execute("test", read_pages=2)
        content = result["content"]
        assert "page text" in content
        assert "Page: http://a.com" in content
        assert "Page: http://b.com" in content
        assert result["metadata"]["engines"] == ["duckduckgo"]

    @pytest.mark.asyncio
    async def test_bing_fallback_when_ddg_empty(self):
        tool = WebSearchTool()
        with patch("agent.tools.builtin_tools.web_search.duckduckgo_search", return_value=[]):
            with patch("agent.tools.builtin_tools.web_search.bing_search", return_value=[
                {"title": "Bing Result", "url": "http://bing.com", "snippet": "bing snippet"},
            ]):
                result = await tool.execute("test")
        assert "Bing Result" in result["content"]
        assert "bing.com" in result["content"]
        assert result["metadata"]["engines"] == ["bing"]

    @pytest.mark.asyncio
    async def test_merges_ddg_and_bing_results(self):
        """Parallel engines merge with canonical-URL deduplication."""
        tool = WebSearchTool()
        ddg = [
            {"title": "DDG One", "url": "https://example.com/a", "snippet": "s1"},
            {"title": "DDG Two", "url": "https://other.com/b", "snippet": "s2"},
        ]
        bing = [
            {"title": "Bing One", "url": "https://www.example.com/a?utm_source=x", "snippet": "s3"},
            {"title": "Bing Two", "url": "https://third.com/c", "snippet": "s4"},
        ]
        with patch("agent.tools.builtin_tools.web_search.duckduckgo_search", return_value=ddg), \
             patch("agent.tools.builtin_tools.web_search.bing_search", return_value=bing):
            result = await tool.execute("q-merge", results=5)
        content = result["content"]
        # example.com/a duplicated via canonical form — appears only once
        assert content.count("example.com/a") == 1
        assert "other.com/b" in content
        assert "third.com/c" in content
        assert set(result["metadata"]["engines"]) == {"duckduckgo", "bing"}

    @pytest.mark.asyncio
    async def test_all_engines_fail_raises(self):
        """Total engine failure surfaces an error instead of an empty list."""
        from agent.tools.builtin_tools.common import SearchEngineError
        tool = WebSearchTool()
        with patch(
            "agent.tools.builtin_tools.web_search.duckduckgo_search",
            side_effect=SearchEngineError("DuckDuckGo failed: blocked"),
        ), patch(
            "agent.tools.builtin_tools.web_search.bing_search",
            side_effect=SearchEngineError("Bing failed: blocked"),
        ):
            with pytest.raises(RuntimeError, match="Web search unavailable"):
                await tool.execute("q-fail")

    @pytest.mark.asyncio
    async def test_no_results_when_engines_empty(self):
        tool = WebSearchTool()
        with patch("agent.tools.builtin_tools.web_search.duckduckgo_search", return_value=[]), \
             patch("agent.tools.builtin_tools.web_search.bing_search", return_value=[]):
            result = await tool.execute("q-none")
        assert "No results found" in result["content"]

    @pytest.mark.asyncio
    async def test_partial_engine_failure_adds_note(self):
        from agent.tools.builtin_tools.common import SearchEngineError
        tool = WebSearchTool()
        with patch(
            "agent.tools.builtin_tools.web_search.duckduckgo_search",
            side_effect=SearchEngineError("blocked"),
        ), patch("agent.tools.builtin_tools.web_search.bing_search", return_value=[
            {"title": "Bing", "url": "https://bing.com/x", "snippet": "s"},
        ]):
            result = await tool.execute("q-partial")
        assert "bing.com/x" in result["content"]
        assert "duckduckgo" in result["content"]
        assert result["metadata"]["errors"]["duckduckgo"]

    @pytest.mark.asyncio
    async def test_serpapi_preferred_when_key_configured(self):
        """With a SERPAPI_API_KEY, auto mode prefers SerpAPI and skips scraping."""
        tool = WebSearchTool()
        serp_results = [{"title": "SERP", "url": "https://serp.example", "snippet": "s"}]
        with patch(
            "agent.tools.builtin_tools.web_search.serpapi_api_key",
            return_value="test-key",
        ), patch(
            "agent.tools.builtin_tools.web_search.serpapi_search",
            new=AsyncMock(return_value=serp_results),
        ) as _:
            with patch("agent.tools.builtin_tools.web_search.duckduckgo_search", new=AsyncMock()) as ddg_mock, \
                 patch("agent.tools.builtin_tools.web_search.bing_search", new=AsyncMock()) as bing_mock:
                result = await tool.execute("q-serp")
        assert "serp.example" in result["content"]
        assert result["metadata"]["engines"] == ["serpapi"]
        ddg_mock.assert_not_called()
        bing_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_read_pages_skips_unfetchable_url(self):
        tool = WebSearchTool()
        with patch("agent.tools.builtin_tools.web_search.duckduckgo_search", return_value=[
            {"title": "Broken", "url": "javascript:void(0)", "snippet": ""},
            {"title": "Good", "url": "https://good.example", "snippet": ""},
        ]), patch("agent.tools.builtin_tools.web_search.bing_search", return_value=[]):
            with patch.object(WebScraperTool, "execute", return_value="text") as scraper_mock:
                result = await tool.execute("q-skip", read_pages=2)
        assert "[Skipped: not a fetchable URL]" in result["content"]
        scraper_mock.assert_called_once()


class TestSearchUrlResolution:
    """resolve_search_url decodes engine redirect links and strips tracking."""

    def _resolve(self, href: str) -> str:
        from agent.tools.builtin_tools.common import resolve_search_url
        return resolve_search_url(href)

    @pytest.mark.asyncio
    async def test_serpapi_tool_reads_api_key_from_tool_config(self):
        """tool_config.api_key must reach serpapi_search — before this it was
        silently ignored and only the env var worked."""
        from agent.tools.builtin_tools.serpapi_search import SerpApiSearchTool

        tool = SerpApiSearchTool()
        with patch(
            "agent.tools.builtin_tools.serpapi_search.serpapi_search",
            new=AsyncMock(return_value=[{"title": "t", "url": "u", "snippet": "s"}]),
        ) as mock_search:
            result = await tool.execute("q", context={"tool_config": {"api_key": "k123"}})
        assert mock_search.await_args.kwargs["api_key"] == "k123"
        assert "t" in result

    @pytest.mark.asyncio
    async def test_webhook_allowed_hosts_glob(self):
        from agent.tools.builtin_tools.webhook_post import WebhookPostTool

        tool = WebhookPostTool()
        ctx = {"tool_config": {"allowed_hosts": ["*.example.org"]}}
        with pytest.raises(ValueError, match="allowed_hosts"):
            await tool.execute("https://evil.net/hook", context=ctx)
        # allowed host reaches the request path (fails later on connect)
        with patch("aiohttp.ClientSession") as sess:
            sess.return_value.__aenter__ = AsyncMock(
                side_effect=RuntimeError("boom")
            )
            with pytest.raises(RuntimeError):
                await tool.execute(
                    "https://api.example.org/hook", context=ctx
                )

    @pytest.mark.asyncio
    async def test_webhook_private_ip_blocked(self):
        from agent.tools.builtin_tools.webhook_post import WebhookPostTool

        tool = WebhookPostTool()
        ctx = {"tool_config": {"allow_private_ips": False}}
        with pytest.raises(ValueError, match="non-public"):
            await tool.execute("https://192.168.1.5/hook", context=ctx)
        with pytest.raises(ValueError, match="non-public"):
            await tool.execute("https://169.254.169.254/latest", context=ctx)
        # IPv6 literals carry []-brackets — naive string splitting would
        # parse "[" as the host and bypass the guard entirely.
        with pytest.raises(ValueError, match="non-public"):
            await tool.execute("https://[::1]/hook", context=ctx)
        with pytest.raises(ValueError, match="non-public"):
            await tool.execute("https://[fd00::1]:8443/hook", context=ctx)
        # IPv6 literals must also match allowed_hosts globs by address.
        ctx2 = {"tool_config": {"allowed_hosts": ["*.example.org"]}}
        with pytest.raises(ValueError, match="allowed_hosts"):
            await tool.execute("https://[2001:db8::1]/hook", context=ctx2)

    def test_ddg_uddg_redirect(self):
        href = (
            '//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.example.com%2F'
            '&amp;rut=abc123'
        )
        assert self._resolve(href) == "https://www.example.com/"

    def test_bing_ck_redirect_base64(self):
        import base64
        target_b64 = base64.b64encode(b"https://example.com/page").decode().rstrip("=")
        href = f'https://www.bing.com/ck/a?p=x&amp;u=a1{target_b64}&amp;ntb=1'
        assert self._resolve(href) == "https://example.com/page"

    def test_bing_ck_without_target_dropped(self):
        assert self._resolve('https://www.bing.com/ck/a?p=x&ntb=1') == ""

    def test_tracking_params_stripped(self):
        url = "https://example.com/page?utm_source=news&fbclid=abc&id=7"
        assert self._resolve(url) == "https://example.com/page?id=7"

    def test_scheme_relative_non_ddg(self):
        assert self._resolve("//cdn.example.com/x") == "https://cdn.example.com/x"

    def test_non_http_dropped(self):
        assert self._resolve("javascript:void(0)") == ""
        assert self._resolve("/relative/path") == ""
        assert self._resolve("") == ""

    def test_canonical_url_dedup_form(self):
        from agent.tools.builtin_tools.common import canonical_url
        assert canonical_url("https://www.Example.com/page/?utm_x=1#frag") == \
            canonical_url("https://example.com/page")


class TestDuckDuckGoParsing:
    """Parser fixtures matching the real DDG markup (rel/class/href order + uddg)."""

    _DDG_HTML = """
    <div class="result results_links">
      <h2 class="result__title">
        <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.example.com%2F&amp;rut=aaaa">Example University</a>
      </h2>
      <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.example.com%2F&amp;rut=aaaa">Die <b>Example</b> University ist eine Universität.</a>
    </div>
    <div class="result results_links">
      <h2 class="result__title">
        <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fno-snippet&amp;rut=bbbb">No Snippet Result</a>
      </h2>
    </div>
    <div class="result results_links">
      <h2 class="result__title">
        <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fthird.example%2F&amp;rut=cccc">Third &amp; Co</a>
      </h2>
      <a class="result__snippet">Third snippet text</a>
    </div>
    """

    def test_parse_ddg_resolves_uddg_and_entities(self):
        from agent.tools.builtin_tools.common import parse_duckduckgo_results
        results = parse_duckduckgo_results(self._DDG_HTML, max_results=5)
        assert len(results) == 3
        assert results[0]["url"] == "https://www.example.com/"
        assert results[0]["title"] == "Example University"
        assert "Example University ist eine Universität" in results[0]["snippet"]

    def test_parse_ddg_missing_snippet_does_not_shift(self):
        """A result without a snippet must not steal the next result's snippet."""
        from agent.tools.builtin_tools.common import parse_duckduckgo_results
        results = parse_duckduckgo_results(self._DDG_HTML, max_results=5)
        assert results[1]["url"] == "https://example.org/no-snippet"
        assert results[1]["snippet"] == ""
        assert results[2]["snippet"] == "Third snippet text"

    def test_parse_ddg_entities_in_title(self):
        from agent.tools.builtin_tools.common import parse_duckduckgo_results
        results = parse_duckduckgo_results(self._DDG_HTML, max_results=5)
        assert results[2]["title"] == "Third & Co"

    def test_parse_ddg_lite_markup(self):
        from agent.tools.builtin_tools.common import parse_duckduckgo_lite_results
        html = """
        <table>
          <tr><td><a rel="nofollow" class="result-link" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Flite.example%2F&amp;rut=x">Lite Result</a></td></tr>
          <tr><td class="result-snippet">Lite snippet</td></tr>
        </table>
        """
        results = parse_duckduckgo_lite_results(html, max_results=5)
        assert len(results) == 1
        assert results[0]["url"] == "https://lite.example/"
        assert results[0]["snippet"] == "Lite snippet"


class TestMergeSearchResults:
    def test_domain_diversity_cap(self):
        from agent.tools.builtin_tools.common import merge_search_results
        ddg = [
            {"title": f"a{i}", "url": f"https://same.com/p{i}", "snippet": ""}
            for i in range(4)
        ]
        bing = [{"title": "b", "url": "https://other.com/x", "snippet": ""}]
        merged = merge_search_results(
            [("duckduckgo", ddg), ("bing", bing)], max_results=4
        )
        same_count = sum(1 for r in merged if "same.com" in r["url"])
        # cap allows 2/domain, relaxed pass fills remaining slots
        assert same_count <= 4
        assert any("other.com" in r["url"] for r in merged)
        assert merged[0]["engine"] == "duckduckgo"

    def test_interleaves_by_rank(self):
        from agent.tools.builtin_tools.common import merge_search_results
        ddg = [{"title": "d1", "url": "https://d1.com", "snippet": ""}]
        bing = [
            {"title": "b1", "url": "https://b1.com", "snippet": ""},
            {"title": "b2", "url": "https://b2.com", "snippet": ""},
        ]
        merged = merge_search_results([("duckduckgo", ddg), ("bing", bing)], 3)
        assert [r["url"] for r in merged] == [
            "https://d1.com", "https://b1.com", "https://b2.com",
        ]


class TestBingParsing:
    def test_parse_bing_results(self):
        import agent.tools.builtin as builtin_mod
        html = (
            '<li class="b_algo">'
            '<h2><a href="https://example.com/page">Example Title</a></h2>'
            '<p>Example snippet text</p>'
            '</li>'
        )
        results = builtin_mod._parse_bing_results(html, max_results=5)
        assert len(results) == 1
        assert results[0]["title"] == "Example Title"
        assert results[0]["url"] == "https://example.com/page"
        assert results[0]["snippet"] == "Example snippet text"

    def test_parse_bing_loose_fallback(self):
        import agent.tools.builtin as builtin_mod
        html = (
            '<h2><a href="https://example.com/page">Loose Title</a></h2>'
        )
        results = builtin_mod._parse_bing_results(html, max_results=5)
        assert len(results) == 1
        assert results[0]["title"] == "Loose Title"
        assert results[0]["url"] == "https://example.com/page"

    def test_parse_bing_empty(self):
        import agent.tools.builtin as builtin_mod
        results = builtin_mod._parse_bing_results("<html></html>", max_results=5)
        assert len(results) == 0

    def test_parse_bing_resolves_ck_redirect(self):
        """Bing /ck/a tracker links must be decoded to the real destination."""
        import agent.tools.builtin as builtin_mod
        import base64
        target_b64 = base64.b64encode(b"https://example.com/real").decode().rstrip("=")
        html = (
            '<li class="b_algo">'
            f'<h2><a href="https://www.bing.com/ck/a?u=a1{target_b64}&amp;ntb=1">Example Title</a></h2>'
            '<p>Snippet &amp; text</p>'
            '</li>'
        )
        results = builtin_mod._parse_bing_results(html, max_results=5)
        assert len(results) == 1
        assert results[0]["url"] == "https://example.com/real"
        assert results[0]["snippet"] == "Snippet & text"

    def test_parse_bing_unescapes_entities(self):
        import agent.tools.builtin as builtin_mod
        html = (
            '<li class="b_algo">'
            '<h2><a href="https://example.com">Example University &#8211; Uni</a></h2>'
            '<p>Example University &amp; Research</p>'
            '</li>'
        )
        results = builtin_mod._parse_bing_results(html, max_results=5)
        assert results[0]["title"] == "Example University – Uni"
        assert results[0]["snippet"] == "Example University & Research"


class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_fetch_retry_429_then_200(self):
        import agent.tools.builtin as builtin_mod
        orig_max = builtin_mod._RETRY_MAX_ATTEMPTS
        builtin_mod._RETRY_MAX_ATTEMPTS = 2
        try:
            resp_429 = MagicMock()
            resp_429.status = 429
            resp_200 = MagicMock()
            resp_200.status = 200
            resp_200.headers = {"Content-Type": "text/plain"}
            resp_200.text = AsyncMock(return_value="success")

            responses = iter([resp_429, resp_200])

            class FakeResponse:
                def __init__(self, resp):
                    self._resp = resp
                async def __aenter__(self):
                    return self._resp
                async def __aexit__(self, *args):
                    return False

            class FakeSession:
                def __init__(self, *args, **kwargs):
                    pass
                async def __aenter__(self):
                    return self
                async def __aexit__(self, *args):
                    return False
                def get(self, *args, **kwargs):
                    return FakeResponse(next(responses))

            with patch("aiohttp.ClientSession", FakeSession):
                result = await builtin_mod._fetch_with_retry("http://example.com")
            assert result == "success"
        finally:
            builtin_mod._RETRY_MAX_ATTEMPTS = orig_max


class TestCaching:
    @pytest.mark.asyncio
    async def test_cache_hit(self):
        import agent.tools.builtin as builtin_mod
        tool = WebScraperTool()
        cache_key = hashlib.sha256("http://cached.example:False".encode()).hexdigest()
        builtin_mod._url_cache[cache_key] = (asyncio.get_running_loop().time(), "cached content")
        try:
            result = await tool.execute("http://cached.example")
            assert result == "cached content"
        finally:
            builtin_mod._url_cache.pop(cache_key, None)


class TestJSFallback:
    @pytest.mark.asyncio
    async def test_js_fallback_when_playwright_missing(self):
        tool = WebScraperTool()
        with patch.dict("sys.modules", {"playwright": None, "playwright.async_api": None}):
            with patch("agent.tools.builtin_tools.web_scraper.fetch_with_retry", return_value="fallback text"):
                result = await tool.execute("http://example.com", render_js=True)
        assert result == "fallback text"


class TestOpenAlexSearchTool:
    @pytest.mark.asyncio
    async def test_tool_schema(self):
        tool = OpenAlexSearchTool()
        ts = tool.tool_schema
        assert ts["type"] == "function"
        assert ts["function"]["name"] == "openalex_search"
        props = ts["function"]["parameters"]["properties"]
        assert "fetch_pdfs" in props
        assert props["fetch_pdfs"]["type"] == "boolean"

    @pytest.mark.asyncio
    async def test_execute_works_mocked(self):
        tool = OpenAlexSearchTool()
        mock_data = {
            "results": [
                {
                    "display_name": "Test Paper",
                    "authorships": [
                        {"author": {"display_name": "Alice", "orcid": "https://orcid.org/0000"}}
                    ],
                    "abstract": "This is a test abstract.",
                    "doi": "https://doi.org/10.1234/test",
                    "publication_date": "2023-01-01",
                    "type": "article",
                    "cited_by_count": 42,
                    "open_access": {
                        "is_oa": True,
                        "oa_status": "gold",
                        "oa_url": "https://example.com/pdf",
                    },
                    "topics": [{"display_name": "AI"}],
                    "primary_location": {"source": {"display_name": "Nature"}},
                }
            ]
        }

        class FakeResponse:
            def __init__(self):
                self.status = 200
                self.headers = {}
            async def json(self):
                return mock_data
            async def text(self):
                return ""
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False

        class FakeSession:
            def __init__(self, *args, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False
            def get(self, *args, **kwargs):
                return FakeResponse()

        with patch("aiohttp.ClientSession", FakeSession):
            result = await tool.execute("machine learning", entity="works")
        assert "Test Paper" in result
        assert "Alice" in result
        assert "DOI: https://doi.org/10.1234/test" in result
        assert "Cited by: 42" in result
        assert "Open Access" in result
        assert "Nature" in result

    @pytest.mark.asyncio
    async def test_execute_works_with_pdf(self):
        tool = OpenAlexSearchTool()
        mock_data = {
            "results": [
                {
                    "display_name": "Test Paper",
                    "authorships": [],
                    "open_access": {
                        "is_oa": True,
                        "oa_status": "gold",
                        "oa_url": "https://example.com/paper.pdf",
                    },
                    "primary_location": {"source": {}},
                }
            ]
        }

        class FakeResponse:
            def __init__(self):
                self.status = 200
                self.headers = {}
            async def json(self):
                return mock_data
            async def text(self):
                return ""
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False

        class FakeSession:
            def __init__(self, *args, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False
            def get(self, *args, **kwargs):
                return FakeResponse()

        with patch("aiohttp.ClientSession", FakeSession):
            with patch.object(
                tool, "_download_and_extract_pdf", return_value="Extracted PDF text content"
            ):
                result = await tool.execute("test", entity="works", fetch_pdfs=True)
        assert "Extracted PDF text content" in result
        assert "--- PDF: https://example.com/paper.pdf ---" in result

    @pytest.mark.asyncio
    async def test_pdf_failure(self):
        tool = OpenAlexSearchTool()
        mock_data = {
            "results": [
                {
                    "display_name": "Test Paper",
                    "authorships": [],
                    "open_access": {
                        "is_oa": True,
                        "oa_status": "gold",
                        "oa_url": "https://example.com/paper.pdf",
                    },
                    "primary_location": {"source": {}},
                }
            ]
        }

        class FakeResponse:
            def __init__(self):
                self.status = 200
                self.headers = {}
            async def json(self):
                return mock_data
            async def text(self):
                return ""
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False

        class FakeSession:
            def __init__(self, *args, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False
            def get(self, *args, **kwargs):
                return FakeResponse()

        with patch("aiohttp.ClientSession", FakeSession):
            with patch.object(
                tool, "_download_and_extract_pdf", side_effect=RuntimeError("Download failed")
            ):
                result = await tool.execute("test", entity="works", fetch_pdfs=True)
        assert "[PDF nicht lesbar: https://example.com/paper.pdf" in result

    @pytest.mark.asyncio
    async def test_execute_authors_mocked(self):
        tool = OpenAlexSearchTool()
        mock_data = {
            "results": [
                {
                    "display_name": "Alice Smith",
                    "orcid": "https://orcid.org/0000-0001",
                    "works_count": 50,
                    "cited_by_count": 1000,
                    "last_known_institution": {"display_name": "MIT"},
                }
            ]
        }

        class FakeResponse:
            def __init__(self):
                self.status = 200
                self.headers = {}
            async def json(self):
                return mock_data
            async def text(self):
                return ""
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False

        class FakeSession:
            def __init__(self, *args, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False
            def get(self, *args, **kwargs):
                return FakeResponse()

        with patch("aiohttp.ClientSession", FakeSession):
            result = await tool.execute("Alice", entity="authors")
        assert "Alice Smith" in result
        assert "MIT" in result
        assert "Works: 50" in result
        assert "Cited by: 1000" in result

    @pytest.mark.asyncio
    async def test_empty_results(self):
        tool = OpenAlexSearchTool()
        mock_data = {"results": []}

        class FakeResponse:
            def __init__(self):
                self.status = 200
                self.headers = {}
            async def json(self):
                return mock_data
            async def text(self):
                return ""
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False

        class FakeSession:
            def __init__(self, *args, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False
            def get(self, *args, **kwargs):
                return FakeResponse()

        with patch("aiohttp.ClientSession", FakeSession):
            result = await tool.execute("xyznonexistent", entity="works")
        assert "No results found" in result

    def test_registry_includes_openalex(self):
        registry = ToolRegistry()
        builtins = registry.list_tools(enabled_only=True)
        assert "openalex_search" in builtins

    def test_resolve_pdf_url_arxiv(self):
        tool = OpenAlexSearchTool()
        assert tool._resolve_pdf_url("https://arxiv.org/abs/2101.12345") == "https://arxiv.org/pdf/2101.12345.pdf"
        assert tool._resolve_pdf_url("https://arxiv.org/abs/2101.12345v2") == "https://arxiv.org/pdf/2101.12345v2.pdf"

    def test_resolve_pdf_url_doi(self):
        tool = OpenAlexSearchTool()
        # DOI resolvers should be left as-is (redirects are followed at download time)
        assert tool._resolve_pdf_url("https://doi.org/10.1234/example") == "https://doi.org/10.1234/example"
        assert tool._resolve_pdf_url("https://dx.doi.org/10.5678/test") == "https://dx.doi.org/10.5678/test"

    def test_resolve_pdf_url_unknown(self):
        tool = OpenAlexSearchTool()
        # Unknown URLs are returned unchanged
        assert tool._resolve_pdf_url("https://example.com/paper.pdf") == "https://example.com/paper.pdf"

    @pytest.mark.asyncio
    async def test_download_pdf_content_type_html(self):
        """If server returns HTML Content-Type and no PDF magic bytes, fail with a clear message."""
        tool = OpenAlexSearchTool()

        class FakeResponse:
            def __init__(self):
                self.status = 200
                self.headers = {"Content-Type": "text/html; charset=utf-8"}
            async def read(self):
                return b"<html><body>Not a PDF</body></html>"
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False

        class FakeSession:
            def __init__(self, *args, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False
            def get(self, *args, **kwargs):
                return FakeResponse()

        with patch("aiohttp.ClientSession", FakeSession):
            with pytest.raises(RuntimeError, match="Expected PDF but got Content-Type"):
                await tool._download_and_extract_pdf("https://example.com/paper.pdf")

    @pytest.mark.asyncio
    async def test_download_pdf_magic_bytes_fallback(self):
        """If Content-Type is wrong but magic bytes are %PDF, parsing should proceed."""
        tool = OpenAlexSearchTool()

        class FakeResponse:
            def __init__(self):
                self.status = 200
                self.headers = {"Content-Type": "application/octet-stream"}
            async def read(self):
                # Minimal valid PDF header (pypdf will reject it later, but that's a parse error)
                return b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False

        class FakeSession:
            def __init__(self, *args, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False
            def get(self, *args, **kwargs):
                return FakeResponse()

        with patch("aiohttp.ClientSession", FakeSession):
            # pypdf will fail to parse this stub, but we want to verify it passes
            # the content-type / magic-bytes gate.  Catch the parse error.
            with pytest.raises(RuntimeError, match="Failed to parse PDF"):
                await tool._download_and_extract_pdf("https://example.com/paper.pdf")

    @pytest.mark.asyncio
    async def test_execute_with_api_key_and_mailto_in_context(self):
        tool = OpenAlexSearchTool()
        mock_data = {"results": [{"display_name": "Test Paper", "authorships": [], "primary_location": {"source": {}}}]}
        captured_params: dict = {}

        class FakeResponse:
            def __init__(self):
                self.status = 200
                self.headers = {}
            async def json(self):
                return mock_data
            async def text(self):
                return ""
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False

        class FakeSession:
            def __init__(self, *args, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False
            def get(self, *args, **kwargs):
                captured_params.update(kwargs.get("params", {}))
                return FakeResponse()

        with patch("aiohttp.ClientSession", FakeSession):
            result = await tool.execute(
                "machine learning",
                entity="works",
                context={"tool_config": {"api_key": "my_test_key", "mailto": "test@example.com"}},
            )
        assert "Test Paper" in result
        assert captured_params.get("api_key") == "my_test_key"
        assert captured_params.get("mailto") == "test@example.com"

    @pytest.mark.asyncio
    async def test_execute_without_config_no_error(self):
        tool = OpenAlexSearchTool()
        mock_data = {"results": [{"display_name": "Test Paper", "authorships": [], "primary_location": {"source": {}}}]}
        captured_params: dict = {}

        class FakeResponse:
            def __init__(self):
                self.status = 200
                self.headers = {}
            async def json(self):
                return mock_data
            async def text(self):
                return ""
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False

        class FakeSession:
            def __init__(self, *args, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False
            def get(self, *args, **kwargs):
                captured_params.update(kwargs.get("params", {}))
                return FakeResponse()

        with patch("aiohttp.ClientSession", FakeSession):
            result = await tool.execute("machine learning", entity="works")
        assert "Test Paper" in result
        assert "api_key" not in captured_params
        assert "mailto" not in captured_params

    @pytest.mark.asyncio
    async def test_retry_503_then_success(self):
        tool = OpenAlexSearchTool()
        mock_data = {"results": [{"display_name": "Test Paper", "authorships": [], "primary_location": {"source": {}}}]}
        call_count = 0

        class FakeResponse:
            def __init__(self, status=200):
                self.status = status
                self.headers = {}
            async def json(self):
                return mock_data
            async def text(self):
                return "error"
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False

        class FakeSession:
            def __init__(self, *args, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False
            def get(self, *args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return FakeResponse(503)
                return FakeResponse(200)

        with patch("aiohttp.ClientSession", FakeSession):
            result = await tool.execute("machine learning", entity="works")
        assert "Test Paper" in result
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_429_then_success(self):
        tool = OpenAlexSearchTool()
        mock_data = {"results": [{"display_name": "Test Paper", "authorships": [], "primary_location": {"source": {}}}]}
        call_count = 0

        class FakeResponse:
            def __init__(self, status=200):
                self.status = status
                self.headers = {}
            async def json(self):
                return mock_data
            async def text(self):
                return "rate limited"
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False

        class FakeSession:
            def __init__(self, *args, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False
            def get(self, *args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    return FakeResponse(429)
                return FakeResponse(200)

        with patch("aiohttp.ClientSession", FakeSession):
            result = await tool.execute("machine learning", entity="works")
        assert "Test Paper" in result
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_503(self):
        tool = OpenAlexSearchTool()
        call_count = 0

        class FakeResponse:
            def __init__(self):
                self.status = 503
                self.headers = {}
            async def json(self):
                return {}
            async def text(self):
                return "Application Error"
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False

        class FakeSession:
            def __init__(self, *args, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False
            def get(self, *args, **kwargs):
                nonlocal call_count
                call_count += 1
                return FakeResponse()

        with patch("aiohttp.ClientSession", FakeSession):
            with pytest.raises(RuntimeError, match="OpenAlex request failed after retries"):
                await tool.execute("machine learning", entity="works")
        assert call_count == 4

    @pytest.mark.asyncio
    async def test_immediate_error_not_retried(self):
        tool = OpenAlexSearchTool()
        call_count = 0

        class FakeResponse:
            def __init__(self):
                self.status = 400
                self.headers = {}
            async def json(self):
                return {}
            async def text(self):
                return "Bad Request"
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False

        class FakeSession:
            def __init__(self, *args, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False
            def get(self, *args, **kwargs):
                nonlocal call_count
                call_count += 1
                return FakeResponse()

        with patch("aiohttp.ClientSession", FakeSession):
            with pytest.raises(RuntimeError, match="OpenAlex request failed with HTTP 400"):
                await tool.execute("machine learning", entity="works")
        assert call_count == 1


class TestCollectionManagerTool:
    def _make_tool(self):
        rag_app = MagicMock()
        rag_app.engine = MagicMock()
        # Let RawFileStore queries resolve cleanly (no real DB round-trip).
        _conn = rag_app.engine.begin.return_value.__aenter__.return_value
        _res = MagicMock()
        _res.mappings.return_value.first.return_value = None
        _res.mappings.return_value.all.return_value = []
        _conn.execute = AsyncMock(return_value=_res)
        rag_app.engine.connect.return_value = rag_app.engine.begin.return_value
        rag_app.config.knowledge_base_root = MagicMock()
        rag_app.index_knowledge_files = AsyncMock(return_value={"ok": True})
        rag_app.delete_knowledge_source = AsyncMock(return_value={"deleted": True})
        return CollectionManagerTool(rag_app)

    def _patch_collections(self, tool, collection_names=None):
        """Patch _get_collection_names to return a fixed list."""
        if collection_names is None:
            collection_names = ["test_collection"]
        from unittest.mock import patch
        return patch.object(tool, "_get_collection_names", new=AsyncMock(return_value=collection_names))

    @pytest.mark.asyncio
    async def test_tool_schema(self):
        tool = self._make_tool()
        ts = tool.tool_schema
        assert ts["type"] == "function"
        assert ts["function"]["name"] == "collection_manager"
        props = ts["function"]["parameters"]["properties"]
        assert "action" in props
        assert "collection" in props
        assert "filename" in props
        assert "content" in props
        assert "limit" in props
        assert "offset" in props
        assert "lines_start" in props
        assert "lines_end" in props
        assert "operation" in props
        assert "line" in props
        assert "after_text" in props
        assert "text_snippet" in props
        assert "old_text" in props
        assert props["action"]["enum"] == ["create", "delete", "list", "read", "edit"]
        assert "collection" in ts["function"]["parameters"]["required"]

    @pytest.mark.asyncio
    async def test_refresh_schema_populates_enum(self):
        tool = self._make_tool()
        with self._patch_collections(tool, ["col_a", "col_b"]):
            await tool.refresh_schema()
        ts = tool.tool_schema
        props = ts["function"]["parameters"]["properties"]
        assert props["collection"].get("enum") == ["col_a", "col_b"]
        assert "col_a, col_b" in ts["function"]["description"]

    @pytest.mark.asyncio
    async def test_refresh_schema_empty_collections(self):
        tool = self._make_tool()
        with self._patch_collections(tool, []):
            await tool.refresh_schema()
        ts = tool.tool_schema
        props = ts["function"]["parameters"]["properties"]
        assert "enum" not in props["collection"]
        assert "Configured collections" not in ts["function"]["description"]

    @pytest.mark.asyncio
    async def test_refresh_schema_no_store(self):
        tool = self._make_tool()
        tool._rag_app._tools_store = None
        await tool.refresh_schema()
        assert tool._collection_names == []
        ts = tool.tool_schema
        props = ts["function"]["parameters"]["properties"]
        assert "enum" not in props["collection"]

    @pytest.mark.asyncio
    async def test_no_collections_configured(self):
        tool = self._make_tool()
        with self._patch_collections(tool, []):
            result = await tool.execute("list", collection="test_collection")
        assert "No collections configured" in result

    @pytest.mark.asyncio
    async def test_invalid_collection(self):
        tool = self._make_tool()
        with self._patch_collections(tool, ["allowed_collection"]):
            result = await tool.execute("list", collection="forbidden_collection")
        assert "is not allowed" in result
        assert "allowed_collection" in result

    @pytest.mark.asyncio
    async def test_list_empty(self):
        tool = self._make_tool()
        with self._patch_collections(tool):
            with patch("app.stores.RawFileStore") as MockStore:
                instance = MockStore.return_value
                instance.list_by_collection = AsyncMock(return_value=[])
                result = await tool.execute("list", collection="test_collection")
        assert "no .md files" in result

    @pytest.mark.asyncio
    async def test_list_with_files(self):
        tool = self._make_tool()
        with self._patch_collections(tool):
            with patch("app.stores.RawFileStore") as MockStore:
                instance = MockStore.return_value
                instance.list_by_collection = AsyncMock(return_value=[
                    {"filename": "a.md"},
                    {"filename": "b.txt"},
                    {"filename": "c.md"},
                ])
                instance.get = AsyncMock(return_value={"file_data": b"# content"})
                result = await tool.execute("list", collection="test_collection")
        assert "a.md" in result
        assert "c.md" in result
        assert "b.txt" not in result
        assert "chars" in result
        assert "lines" in result

    @pytest.mark.asyncio
    async def test_list_pattern_filters(self):
        tool = self._make_tool()
        with self._patch_collections(tool):
            with patch("app.stores.RawFileStore") as MockStore:
                instance = MockStore.return_value
                instance.list_by_collection = AsyncMock(return_value=[
                    {"filename": "report-2024.md"},
                    {"filename": "notes.md"},
                    {"filename": "report-2025.md"},
                ])
                instance.get = AsyncMock(return_value={"file_data": b"# x"})
                result = await tool.execute(
                    "list", collection="test_collection", pattern="report"
                )
        assert "report-2024.md" in result
        assert "report-2025.md" in result
        assert "notes.md" not in result
        assert "matching 'report'" in result

    @pytest.mark.asyncio
    async def test_list_capped_and_paged(self):
        """Large listings must respect the hard limit and accept paging."""
        tool = self._make_tool()
        files = [{"filename": f"doc{i:03d}.md"} for i in range(500)]
        with self._patch_collections(tool):
            with patch("app.stores.RawFileStore") as MockStore:
                instance = MockStore.return_value
                instance.list_by_collection = AsyncMock(return_value=files)
                result = await tool.execute("list", collection="test_collection")
        assert len(result) <= 15_400
        assert "truncated" in result or "excerpt" in result
        with self._patch_collections(tool):
            with patch("app.stores.RawFileStore") as MockStore:
                instance = MockStore.return_value
                instance.list_by_collection = AsyncMock(return_value=files)
                page2 = await tool.execute(
                    "list", collection="test_collection", lines_start=30, lines_end=40
                )
        assert "doc030.md" in page2 or "doc029.md" in page2
        assert "showing lines 30-40" in page2

    @pytest.mark.asyncio
    async def test_read_with_pattern(self):
        tool = self._make_tool()
        content = b"intro\nalpha section\nfiller\ntarget line here\nfiller\noutro"
        with self._patch_collections(tool):
            with patch("app.stores.RawFileStore") as MockStore:
                instance = MockStore.return_value
                instance.get = AsyncMock(return_value={"file_data": content})
                result = await tool.execute(
                    "read", collection="test_collection", filename="doc.md",
                    pattern="target",
                )
        assert "4│ target line here" in result
        assert "matches" in result

    @pytest.mark.asyncio
    async def test_create(self):
        tool = self._make_tool()
        with self._patch_collections(tool):
            with patch("app.stores.RawFileStore") as MockStore:
                instance = MockStore.return_value
                instance.save = AsyncMock()
                result = await tool.execute("create", collection="test_collection", filename="doc.md", content="# Hello")
        assert "Saved" in result
        assert "doc.md" in result
        assert "Indexing scheduled" in result
        assert "test_collection" in tool._pending_index
        assert "doc.md" in tool._pending_index["test_collection"]
        assert tool._index_task is not None
        tool._index_task.cancel()

    @pytest.mark.asyncio
    async def test_create_overwrites(self):
        tool = self._make_tool()
        with self._patch_collections(tool):
            with patch("app.stores.RawFileStore") as MockStore:
                instance = MockStore.return_value
                instance.save = AsyncMock()
                result = await tool.execute("create", collection="test_collection", filename="doc.md", content="# Updated")
        assert "Saved" in result
        assert "Indexing scheduled" in result
        assert "doc.md" in tool._pending_index.get("test_collection", set())
        if tool._index_task:
            tool._index_task.cancel()

    @pytest.mark.asyncio
    async def test_delete(self):
        tool = self._make_tool()
        with self._patch_collections(tool):
            with patch("app.stores.RawFileStore") as MockStore:
                instance = MockStore.return_value
                instance.delete = AsyncMock(return_value=True)
                result = await tool.execute("delete", collection="test_collection", filename="doc.md")
        assert "Deleted" in result
        tool._rag_app.delete_knowledge_source.assert_awaited_once_with("test_collection", "doc.md", delete_file=False)

    @pytest.mark.asyncio
    async def test_read_db(self):
        tool = self._make_tool()
        with self._patch_collections(tool):
            with patch("app.stores.RawFileStore") as MockStore:
                instance = MockStore.return_value
                instance.get = AsyncMock(return_value={"file_data": b"# DB content"})
                result = await tool.execute("read", collection="test_collection", filename="doc.md")
        assert "File: test_collection/doc.md" in result
        assert "1│ # DB content" in result

    @pytest.mark.asyncio
    async def test_read_fs_fallback(self):
        tool = self._make_tool()
        with self._patch_collections(tool):
            with patch("app.stores.RawFileStore") as MockStore:
                instance = MockStore.return_value
                instance.get = AsyncMock(return_value=None)
                fs_path = MagicMock()
                fs_path.exists.return_value = True
                fs_path.read_text.return_value = "# FS content"
                fs_path.__truediv__ = MagicMock(return_value=fs_path)
                tool._rag_app.config.knowledge_base_root.__truediv__ = MagicMock(return_value=fs_path)
                result = await tool.execute("read", collection="test_collection", filename="doc.md")
        assert "File: test_collection/doc.md" in result
        assert "1│ # FS content" in result

    @pytest.mark.asyncio
    async def test_read_not_found(self):
        tool = self._make_tool()
        with self._patch_collections(tool):
            with patch("app.stores.RawFileStore") as MockStore:
                instance = MockStore.return_value
                instance.get = AsyncMock(return_value=None)
                fs_path = MagicMock()
                fs_path.exists.return_value = False
                fs_path.__truediv__ = MagicMock(return_value=fs_path)
                tool._rag_app.config.knowledge_base_root.__truediv__ = MagicMock(return_value=fs_path)
                with pytest.raises(FileNotFoundError):
                    await tool.execute("read", collection="test_collection", filename="doc.md")

    @pytest.mark.asyncio
    async def test_invalid_filename(self):
        tool = self._make_tool()
        with self._patch_collections(tool):
            with pytest.raises(ValueError, match="Invalid filename"):
                await tool.execute("create", collection="test_collection", filename="../hack.md", content="x")

    @pytest.mark.asyncio
    async def test_invalid_extension(self):
        tool = self._make_tool()
        with self._patch_collections(tool):
            with pytest.raises(ValueError, match="Only .md files"):
                await tool.execute("create", collection="test_collection", filename="doc.txt", content="x")

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        tool = self._make_tool()
        with self._patch_collections(tool):
            with patch("app.stores.RawFileStore"):
                with pytest.raises(ValueError, match="Unknown action"):
                    await tool.execute("unknown", collection="test_collection", filename="doc.md")

    @pytest.mark.asyncio
    async def test_read_with_limit(self):
        tool = self._make_tool()
        with self._patch_collections(tool):
            with patch("app.stores.RawFileStore") as MockStore:
                instance = MockStore.return_value
                instance.get = AsyncMock(return_value={"file_data": b"Hello World"})
                result = await tool.execute("read", collection="test_collection", filename="doc.md", limit=5)
        assert "1│ Hello" in result
        assert "truncated" in result
        assert "cut inside long line" in result

    @pytest.mark.asyncio
    async def test_read_with_offset(self):
        tool = self._make_tool()
        with self._patch_collections(tool):
            with patch("app.stores.RawFileStore") as MockStore:
                instance = MockStore.return_value
                instance.get = AsyncMock(return_value={"file_data": b"Hello World"})
                result = await tool.execute("read", collection="test_collection", filename="doc.md", offset=6)
        assert "1│ World" in result
        assert "excerpt" in result
        assert "end of document" in result

    @pytest.mark.asyncio
    async def test_read_with_lines(self):
        tool = self._make_tool()
        with self._patch_collections(tool):
            with patch("app.stores.RawFileStore") as MockStore:
                instance = MockStore.return_value
                instance.get = AsyncMock(return_value={"file_data": b"line1\nline2\nline3\nline4"})
                result = await tool.execute("read", collection="test_collection", filename="doc.md", lines_start=2, lines_end=3)
        assert "2│ line2" in result
        assert "3│ line3" in result
        assert "showing lines 2-3 of 4" in result
        assert "continue with lines_start=4" in result

    @pytest.mark.asyncio
    async def test_edit_insert_at_line(self):
        tool = self._make_tool()
        with self._patch_collections(tool):
            with patch("app.stores.RawFileStore") as MockStore:
                instance = MockStore.return_value
                instance.get = AsyncMock(return_value={"file_data": b"a\nb\nc\n"})
                instance.save = AsyncMock()
                result = await tool.execute("edit", collection="test_collection", filename="doc.md", operation="insert", line=2, content="X\n")
        assert "Edited" in result
        assert "Indexing scheduled" in result
        saved_content = instance.save.await_args.kwargs["data"]
        assert saved_content == b"a\nX\nb\nc\n"
        if tool._index_task:
            tool._index_task.cancel()

    @pytest.mark.asyncio
    async def test_edit_insert_after_text(self):
        tool = self._make_tool()
        with self._patch_collections(tool):
            with patch("app.stores.RawFileStore") as MockStore:
                instance = MockStore.return_value
                instance.get = AsyncMock(return_value={"file_data": b"hello world"})
                instance.save = AsyncMock()
                result = await tool.execute("edit", collection="test_collection", filename="doc.md", operation="insert", after_text="hello ", content="X ")
        assert "Edited" in result
        assert "Indexing scheduled" in result
        saved_content = instance.save.await_args.kwargs["data"]
        assert saved_content == b"hello X world"
        if tool._index_task:
            tool._index_task.cancel()

    @pytest.mark.asyncio
    async def test_edit_remove_lines(self):
        tool = self._make_tool()
        with self._patch_collections(tool):
            with patch("app.stores.RawFileStore") as MockStore:
                instance = MockStore.return_value
                instance.get = AsyncMock(return_value={"file_data": b"a\nb\nc\nd\n"})
                instance.save = AsyncMock()
                result = await tool.execute("edit", collection="test_collection", filename="doc.md", operation="remove", lines_start=2, lines_end=3)
        assert "Edited" in result
        assert "Indexing scheduled" in result
        saved_content = instance.save.await_args.kwargs["data"]
        assert saved_content == b"a\nd\n"
        if tool._index_task:
            tool._index_task.cancel()

    @pytest.mark.asyncio
    async def test_edit_remove_snippet(self):
        tool = self._make_tool()
        with self._patch_collections(tool):
            with patch("app.stores.RawFileStore") as MockStore:
                instance = MockStore.return_value
                instance.get = AsyncMock(return_value={"file_data": b"hello world foo"})
                instance.save = AsyncMock()
                result = await tool.execute("edit", collection="test_collection", filename="doc.md", operation="remove", text_snippet="world ")
        assert "Edited" in result
        assert "Indexing scheduled" in result
        saved_content = instance.save.await_args.kwargs["data"]
        assert saved_content == b"hello foo"
        if tool._index_task:
            tool._index_task.cancel()

    @pytest.mark.asyncio
    async def test_edit_replace(self):
        tool = self._make_tool()
        with self._patch_collections(tool):
            with patch("app.stores.RawFileStore") as MockStore:
                instance = MockStore.return_value
                instance.get = AsyncMock(return_value={"file_data": b"hello world world"})
                instance.save = AsyncMock()
                result = await tool.execute("edit", collection="test_collection", filename="doc.md", operation="replace", old_text="world", content="universe")
        assert "Edited" in result
        assert "Indexing scheduled" in result
        saved_content = instance.save.await_args.kwargs["data"]
        assert saved_content == b"hello universe world"
        if tool._index_task:
            tool._index_task.cancel()

    @pytest.mark.asyncio
    async def test_edit_unknown_operation(self):
        tool = self._make_tool()
        with self._patch_collections(tool):
            with patch("app.stores.RawFileStore") as MockStore:
                instance = MockStore.return_value
                instance.get = AsyncMock(return_value={"file_data": b"content"})
                with pytest.raises(ValueError, match="Unknown edit operation"):
                    await tool.execute("edit", collection="test_collection", filename="doc.md", operation="foo")

    @pytest.mark.asyncio
    async def test_edit_insert_snippet_not_found(self):
        tool = self._make_tool()
        with self._patch_collections(tool):
            with patch("app.stores.RawFileStore") as MockStore:
                instance = MockStore.return_value
                instance.get = AsyncMock(return_value={"file_data": b"content"})
                with pytest.raises(ValueError, match="Snippet not found"):
                    await tool.execute("edit", collection="test_collection", filename="doc.md", operation="insert", after_text="missing", content="x")

    @pytest.mark.asyncio
    async def test_edit_replace_not_found(self):
        tool = self._make_tool()
        with self._patch_collections(tool):
            with patch("app.stores.RawFileStore") as MockStore:
                instance = MockStore.return_value
                instance.get = AsyncMock(return_value={"file_data": b"content"})
                with pytest.raises(ValueError, match="Text to replace not found"):
                    await tool.execute("edit", collection="test_collection", filename="doc.md", operation="replace", old_text="missing", content="x")

    @pytest.mark.asyncio
    async def test_schedule_index_creates_task(self):
        tool = self._make_tool()
        tool._schedule_index("col_a", "file1.md")
        assert "col_a" in tool._pending_index
        assert "file1.md" in tool._pending_index["col_a"]
        assert tool._index_task is not None
        tool._index_task.cancel()

    @pytest.mark.asyncio
    async def test_debounce_cancels_old_task(self):
        import asyncio
        tool = self._make_tool()
        tool._schedule_index("col_a", "file1.md")
        old_task = tool._index_task
        assert old_task is not None
        tool._schedule_index("col_a", "file2.md")
        new_task = tool._index_task
        assert new_task is not old_task
        await asyncio.sleep(0)  # let the event loop process cancellation
        assert old_task.cancelled()
        new_task.cancel()

    @pytest.mark.asyncio
    async def test_run_index_debounced(self):
        import asyncio
        tool = self._make_tool()
        tool._pending_index = {"col_a": {"file1.md", "file2.md"}}
        tool._rag_app.index_knowledge_files = AsyncMock()
        with patch.object(asyncio, "sleep", new=AsyncMock()) as mock_sleep:
            await tool._run_index_debounced()
            mock_sleep.assert_awaited_once_with(60)
        tool._rag_app.index_knowledge_files.assert_awaited_once_with("col_a", ["file1.md", "file2.md"])
        assert not tool._pending_index

    @pytest.mark.asyncio
    async def test_run_index_debounced_custom_delay(self):
        import asyncio
        tool = self._make_tool()
        tool._index_debounce_seconds = 30
        tool._pending_index = {"col_a": {"file1.md"}}
        tool._rag_app.index_knowledge_files = AsyncMock()
        with patch.object(asyncio, "sleep", new=AsyncMock()) as mock_sleep:
            await tool._run_index_debounced()
            mock_sleep.assert_awaited_once_with(30)
        tool._rag_app.index_knowledge_files.assert_awaited_once_with("col_a", ["file1.md"])

    @pytest.mark.asyncio
    async def test_refresh_schema_reads_debounce(self):
        tool = self._make_tool()
        with patch.object(tool, "_get_tool_config", new=AsyncMock(return_value={"index_debounce_seconds": 120})):
            await tool.refresh_schema()
        assert tool._index_debounce_seconds == 120


class TestRssFeedReaderTool:
    @pytest.mark.asyncio
    async def test_tool_schema(self):
        tool = RssFeedReaderTool()
        ts = tool.tool_schema
        assert ts["type"] == "function"
        assert ts["function"]["name"] == "rss_feed_reader"

    def _make_mock_session(self):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.text = AsyncMock(return_value="<feed></feed>")
        mock_get = MagicMock()
        mock_get.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_get.__aexit__ = AsyncMock(return_value=False)
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_get)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        return mock_session

    @pytest.mark.asyncio
    async def test_execute_mocked(self):
        tool = RssFeedReaderTool()
        fake_parsed = MagicMock()
        fake_parsed.entries = [
            {
                "title": "Entry 1",
                "link": "http://example.com/1",
                "published": "2024-01-01",
                "summary": "Summary 1",
            },
            {
                "title": "Entry 2",
                "link": "http://example.com/2",
                "published": "2024-01-02",
                "summary": "Summary 2",
            },
        ]
        fake_parsed.feed = {"title": "Test Feed"}
        fake_parsed.get = MagicMock(return_value=False)

        with patch("agent.tools.builtin_tools.rss_feed_reader.feedparser.parse", return_value=fake_parsed):
            with patch("agent.tools.builtin_tools.rss_feed_reader.aiohttp.ClientSession", return_value=self._make_mock_session()):
                result = await tool.execute("http://example.com/feed.xml", entries=2)

        assert "Test Feed" in result
        assert "Entry 1" in result
        assert "Entry 2" in result
        assert "http://example.com/1" in result
        assert "http://example.com/2" in result

    @pytest.mark.asyncio
    async def test_execute_with_full_content_mocked(self):
        tool = RssFeedReaderTool()
        fake_parsed = MagicMock()
        fake_parsed.entries = [
            {
                "title": "Entry 1",
                "link": "http://example.com/1",
                "published": "2024-01-01",
                "summary": "Summary 1",
            },
        ]
        fake_parsed.feed = {"title": "Test Feed"}
        fake_parsed.get = MagicMock(return_value=False)

        with patch("agent.tools.builtin_tools.rss_feed_reader.feedparser.parse", return_value=fake_parsed):
            with patch("agent.tools.builtin_tools.rss_feed_reader.aiohttp.ClientSession", return_value=self._make_mock_session()):
                with patch.object(WebScraperTool, "execute", return_value="Full article text"):
                    result = await tool.execute(
                        "http://example.com/feed.xml", entries=1, fetch_full_content=True
                    )

        assert "Full article text" in result

    @pytest.mark.asyncio
    async def test_execute_no_entries(self):
        tool = RssFeedReaderTool()
        fake_parsed = MagicMock()
        fake_parsed.entries = []
        fake_parsed.feed = {"title": "Test Feed"}
        fake_parsed.get = MagicMock(return_value=False)

        with patch("agent.tools.builtin_tools.rss_feed_reader.feedparser.parse", return_value=fake_parsed):
            with patch("agent.tools.builtin_tools.rss_feed_reader.aiohttp.ClientSession", return_value=self._make_mock_session()):
                result = await tool.execute("http://example.com/feed.xml", entries=5)

        assert "No entries found" in result


class TestCustomToolWrapperDockerFallback:
    @pytest.mark.asyncio
    async def test_docker_unavailable_falls_back_to_venv(self):
        """When Docker is unavailable, CustomToolWrapper should fallback to venv execution."""
        from agent.tools import CustomToolWrapper
        from agent.tools.venv_executor import VenvExecutor

        mock_entry = MagicMock()
        mock_entry.name = "docker_tool"
        mock_entry.enabled = True
        mock_entry.runtime_mode = "docker"
        mock_entry.code = "def run(**params): return params['x'] * 2"
        mock_entry.tool_schema = {}
        mock_entry.venv_config = {}
        mock_entry.id = "test-docker-tool"

        wrapper = CustomToolWrapper(mock_entry)

        # Mock VenvExecutor.execute to return success
        with patch.object(
            VenvExecutor, "execute", return_value={"ok": True, "content": "84", "error": None}
        ) as mock_venv_exec:
            with patch(
                "agent.tools.docker_sandbox.ConversationSandboxManager.get_or_create",
                side_effect=RuntimeError("Docker sandbox unavailable"),
            ):
                result = await wrapper.execute(x=42, context={"conversation_id": "conv-123"})

        assert result == "84"
        mock_venv_exec.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_conversation_id_falls_back_to_venv(self):
        """When no conversation_id is provided, CustomToolWrapper should fallback to venv execution."""
        from agent.tools import CustomToolWrapper
        from agent.tools.venv_executor import VenvExecutor

        mock_entry = MagicMock()
        mock_entry.name = "docker_tool"
        mock_entry.enabled = True
        mock_entry.runtime_mode = "docker"
        mock_entry.code = "def run(**params): return params['x'] * 2"
        mock_entry.tool_schema = {}
        mock_entry.venv_config = {}
        mock_entry.id = "test-docker-tool"

        wrapper = CustomToolWrapper(mock_entry)

        with patch.object(
            VenvExecutor, "execute", return_value={"ok": True, "content": "84", "error": None}
        ) as mock_venv_exec:
            result = await wrapper.execute(x=42)

        assert result == "84"
        mock_venv_exec.assert_awaited_once()


class TestAgentMemoryTool:
    @pytest.fixture
    def fake_store(self):
        store = MagicMock()
        store.get_memory = AsyncMock(return_value="existing memory")
        store.set_memory = AsyncMock(return_value="new memory")
        store.append_memory = AsyncMock(return_value="existing memory\nappended")
        store.delete_memory = AsyncMock(return_value=True)
        return store

    @pytest.mark.asyncio
    async def test_read_with_memory(self, fake_store):
        tool = AgentMemoryTool()
        result = await tool.execute("read", context={"agent_memory_store": fake_store, "conversation_id": "c1", "assistant_name": "A1"})
        assert "existing memory" in result
        fake_store.get_memory.assert_awaited_once_with("c1", "A1")

    @pytest.mark.asyncio
    async def test_read_empty(self, fake_store):
        fake_store.get_memory = AsyncMock(return_value=None)
        tool = AgentMemoryTool()
        result = await tool.execute("read", context={"agent_memory_store": fake_store, "conversation_id": "c1", "assistant_name": "A1"})
        assert "empty" in result.lower()

    @pytest.mark.asyncio
    async def test_set(self, fake_store):
        tool = AgentMemoryTool()
        result = await tool.execute("set", content="new memory", context={"agent_memory_store": fake_store, "conversation_id": "c1", "assistant_name": "A1"})
        assert "updated" in result.lower()
        fake_store.set_memory.assert_awaited_once_with("c1", "A1", "new memory")

    @pytest.mark.asyncio
    async def test_append(self, fake_store):
        tool = AgentMemoryTool()
        result = await tool.execute("append", content="appended", context={"agent_memory_store": fake_store, "conversation_id": "c1", "assistant_name": "A1"})
        assert "appended" in result.lower()
        fake_store.append_memory.assert_awaited_once_with("c1", "A1", "appended")

    @pytest.mark.asyncio
    async def test_delete(self, fake_store):
        tool = AgentMemoryTool()
        result = await tool.execute("delete", context={"agent_memory_store": fake_store, "conversation_id": "c1", "assistant_name": "A1"})
        assert "cleared" in result.lower()
        fake_store.delete_memory.assert_awaited_once_with("c1", "A1")

    @pytest.mark.asyncio
    async def test_missing_store_raises(self):
        tool = AgentMemoryTool()
        with pytest.raises(RuntimeError, match="AgentMemoryStore not available"):
            await tool.execute("read", context={"conversation_id": "c1", "assistant_name": "A1"})

    @pytest.mark.asyncio
    async def test_missing_conversation_id_raises(self, fake_store):
        tool = AgentMemoryTool()
        with pytest.raises(RuntimeError, match="conversation_id required"):
            await tool.execute("read", context={"agent_memory_store": fake_store, "assistant_name": "A1"})

    @pytest.mark.asyncio
    async def test_set_without_content_raises(self, fake_store):
        tool = AgentMemoryTool()
        with pytest.raises(ValueError, match="content is required"):
            await tool.execute("set", context={"agent_memory_store": fake_store, "conversation_id": "c1", "assistant_name": "A1"})


class TestSSHTool:
    @pytest.mark.asyncio
    async def test_tool_schema(self):
        tool = SSHTool()
        ts = tool.tool_schema
        assert ts["type"] == "function"
        assert ts["function"]["name"] == "ssh_exec"
        assert "command" in ts["function"]["parameters"]["properties"]
        assert "timeout" in ts["function"]["parameters"]["properties"]

    @pytest.mark.asyncio
    async def test_missing_host_raises(self):
        tool = SSHTool()
        with pytest.raises(ValueError, match="host is not configured"):
            await tool.execute("ls", context={"tool_config": {"username": "root", "auth_type": "password"}})

    @pytest.mark.asyncio
    async def test_missing_username_raises(self):
        tool = SSHTool()
        with pytest.raises(ValueError, match="username is not configured"):
            await tool.execute("ls", context={"tool_config": {"host": "example.com", "auth_type": "password"}})

    @pytest.mark.asyncio
    async def test_invalid_auth_type_raises(self):
        tool = SSHTool()
        with pytest.raises(ValueError, match="auth_type must be"):
            await tool.execute("ls", context={"tool_config": {"host": "example.com", "username": "root", "auth_type": "invalid"}})

    @pytest.mark.asyncio
    async def test_password_auth_missing_password_raises(self):
        tool = SSHTool()
        with pytest.raises(ValueError, match="password is not configured"):
            await tool.execute("ls", context={"tool_config": {"host": "example.com", "username": "root", "auth_type": "password"}})

    @pytest.mark.asyncio
    async def test_key_auth_missing_key_raises(self):
        tool = SSHTool()
        with pytest.raises(ValueError, match="private_key is not configured"):
            await tool.execute("ls", context={"tool_config": {"host": "example.com", "username": "root", "auth_type": "key"}})

    @pytest.mark.asyncio
    async def test_execute_password_auth_success(self):
        tool = SSHTool()
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.stdout = "hello\n"
        mock_result.stderr = ""
        mock_result.exit_status = 0
        mock_conn.run = AsyncMock(return_value=mock_result)

        mock_asyncssh = MagicMock()
        mock_asyncssh.connect = MagicMock()
        mock_asyncssh.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_asyncssh.connect.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_asyncssh.PermissionDenied = Exception
        mock_asyncssh.ConnectionLost = Exception
        mock_asyncssh.HostKeyNotVerifiable = Exception
        mock_asyncssh.Error = Exception

        with patch.dict("sys.modules", {"asyncssh": mock_asyncssh}):
            result = await tool.execute(
                "echo hello",
                context={"tool_config": {"host": "example.com", "username": "root", "auth_type": "password", "password": "secret"}}
            )
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_execute_with_stderr_and_exit_code(self):
        tool = SSHTool()
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.stdout = "output"
        mock_result.stderr = "error msg"
        mock_result.exit_status = 1
        mock_conn.run = AsyncMock(return_value=mock_result)

        mock_asyncssh = MagicMock()
        mock_asyncssh.connect = MagicMock()
        mock_asyncssh.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_asyncssh.connect.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_asyncssh.PermissionDenied = Exception
        mock_asyncssh.ConnectionLost = Exception
        mock_asyncssh.HostKeyNotVerifiable = Exception
        mock_asyncssh.Error = Exception

        with patch.dict("sys.modules", {"asyncssh": mock_asyncssh}):
            result = await tool.execute(
                "false",
                context={"tool_config": {"host": "example.com", "username": "root", "auth_type": "password", "password": "secret"}}
            )
        assert "output" in result
        assert "[stderr]" in result
        assert "error msg" in result
        assert "[exit code: 1]" in result

    @pytest.mark.asyncio
    async def test_output_truncation(self):
        tool = SSHTool()
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.stdout = "x" * 20_000
        mock_result.stderr = ""
        mock_result.exit_status = 0
        mock_conn.run = AsyncMock(return_value=mock_result)

        mock_asyncssh = MagicMock()
        mock_asyncssh.connect = MagicMock()
        mock_asyncssh.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_asyncssh.connect.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_asyncssh.PermissionDenied = Exception
        mock_asyncssh.ConnectionLost = Exception
        mock_asyncssh.HostKeyNotVerifiable = Exception
        mock_asyncssh.Error = Exception

        with patch.dict("sys.modules", {"asyncssh": mock_asyncssh}):
            result = await tool.execute(
                "cat bigfile",
                context={"tool_config": {"host": "example.com", "username": "root", "auth_type": "password", "password": "secret"}}
            )
        assert "[output truncated]" in result
        assert len(result) <= 15_100


class TestAssistantCallerTool:
    def test_tool_schema_generated(self):
        from agent.tools.builtin_tools.assistant_caller import AssistantCallerTool
        rag_app = MagicMock()
        tool = AssistantCallerTool(rag_app=rag_app, assistant_name="Test Assistant", description="A test assistant.")
        schema = tool.tool_schema
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "assistant__test_assistant"
        assert "A test assistant." in schema["function"]["description"]
        props = schema["function"]["parameters"]["properties"]
        assert "task" in props
        assert schema["function"]["parameters"]["required"] == ["task"]

    @pytest.mark.asyncio
    async def test_self_call_blocked(self):
        from agent.tools.builtin_tools.assistant_caller import AssistantCallerTool
        rag_app = MagicMock()
        rag_app.ready = True
        tool = AssistantCallerTool(rag_app=rag_app, assistant_name="MyAssistant")
        result = await tool.execute(task="do something", context={"assistant_name": "MyAssistant"})
        assert "cannot call itself" in result

    @pytest.mark.asyncio
    async def test_rag_app_not_ready(self):
        from agent.tools.builtin_tools.assistant_caller import AssistantCallerTool
        rag_app = MagicMock()
        rag_app.ready = False
        tool = AssistantCallerTool(rag_app=rag_app, assistant_name="OtherAssistant")
        result = await tool.execute(task="do something")
        assert "not ready" in result

    @pytest.mark.asyncio
    async def test_successful_call(self):
        from agent.tools.builtin_tools.assistant_caller import AssistantCallerTool
        rag_app = MagicMock()
        rag_app.ready = True
        rag_app.run = AsyncMock(return_value={"output": "sub-answer"})
        tool = AssistantCallerTool(rag_app=rag_app, assistant_name="OtherAssistant")
        result = await tool.execute(task="do something", context={"assistant_name": "Caller"})
        assert result == "sub-answer"
        rag_app.run.assert_awaited_once_with(input_text="do something", model_name="OtherAssistant", chat_history=[])


    def test_refresh_assistant_tools_includes_non_api_accessible(self):
        from agent.application import RAGApplication
        rag_app = MagicMock(spec=RAGApplication)
        rag_app.instructions = {
            "InternalAgent": {
                "description": "An internal agent.",
                "api_accessible": False,
                "agent_enabled": True,
            }
        }
        rag_app._tool_registry = ToolRegistry()
        # Call the real _refresh_assistant_tools method
        RAGApplication._refresh_assistant_tools(rag_app)
        assert rag_app._tool_registry.get_tool("assistant__internalagent") is not None


class TestToolRegistryAssistants:
    def test_register_assistant_tool(self):
        registry = ToolRegistry()
        mock_tool = MagicMock()
        mock_tool.name = "assistant__test"
        registry.register_assistant("assistant__test", mock_tool)
        assert registry.get_tool("assistant__test") is mock_tool

    def test_list_tools_includes_assistant(self):
        registry = ToolRegistry()
        mock_tool = MagicMock()
        mock_tool.name = "assistant__foo"
        registry.register_assistant("assistant__foo", mock_tool)
        tools = registry.list_tools(enabled_only=True)
        assert "assistant__foo" in tools

    def test_clear_assistants(self):
        registry = ToolRegistry()
        mock_tool = MagicMock()
        mock_tool.name = "assistant__foo"
        registry.register_assistant("assistant__foo", mock_tool)
        registry.clear_assistants()
        assert registry.get_tool("assistant__foo") is None

    def test_tool_schemas_include_assistant(self):
        registry = ToolRegistry()
        mock_tool = MagicMock()
        mock_tool.name = "assistant__bar"
        mock_tool.tool_schema = {"type": "function", "function": {"name": "assistant__bar"}}
        registry.register_assistant("assistant__bar", mock_tool)
        schemas = registry.tool_schemas(["assistant__bar"])
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "assistant__bar"


class TestToolCallConflictKeys:
    """Tools that mutate a shared target declare call_conflict_key so the
    runner serializes same-target batch calls instead of racing them."""

    def test_collection_manager_key(self):
        tool = TestCollectionManagerTool()._make_tool()
        for action in ("create", "delete", "edit"):
            key = tool.call_conflict_key(
                {"action": action, "collection": "kb", "filename": "f.md"}
            )
            assert key == "kb/f.md", action
        for action in ("read", "list"):
            assert tool.call_conflict_key(
                {"action": action, "collection": "kb", "filename": "f.md"}
            ) is None, action

    def test_agent_workspace_key(self):
        from agent.tools.builtin_tools.agent_workspace import AgentWorkspaceTool
        tool = AgentWorkspaceTool.__new__(AgentWorkspaceTool)
        for action in ("write", "append", "delete", "move", "mkdir"):
            assert tool.call_conflict_key({"action": action, "path": "a/b.md"}) == "a/b.md", action
        for action in ("read", "list", "inspect", "view"):
            assert tool.call_conflict_key({"action": action, "path": "a/b.md"}) is None, action
        assert tool.call_conflict_key({"action": "list"}) is None

    def test_document_generator_key(self):
        from agent.tools.builtin_tools.document_generator import DocumentGeneratorTool
        tool = DocumentGeneratorTool.__new__(DocumentGeneratorTool)
        assert tool.call_conflict_key({"action": "add_sections", "draft_id": "d1"}) == "draft:d1"
        assert tool.call_conflict_key({"action": "finalize", "draft_id": "d1"}) == "draft:d1"
        assert tool.call_conflict_key({"action": "create", "kind": "docx"}) is None


@pytest.mark.asyncio
async def test_concurrent_collection_edits_both_apply():
    """Two edit calls on the same file must not lose one write: the
    per-file lock forces the second call to re-read after the first
    call's save instead of overwriting a stale snapshot."""
    helper = TestCollectionManagerTool()
    tool = helper._make_tool()
    stored = {"data": b"line1\nline2\nline3\n"}

    with helper._patch_collections(tool):
        with patch("app.stores.RawFileStore") as MockStore:
            instance = MockStore.return_value

            async def _get(collection, source):
                await asyncio.sleep(0)  # yield: without serialization both reads race
                return {"file_data": stored["data"]}

            async def _save(collection_name, source, filename, content_type, data):
                await asyncio.sleep(0)
                stored["data"] = data

            instance.get = _get
            instance.save = _save

            r1, r2 = await asyncio.gather(
                tool.execute(
                    "edit", collection="test_collection", filename="f.md",
                    operation="replace", old_text="line1", content="ONE",
                ),
                tool.execute(
                    "edit", collection="test_collection", filename="f.md",
                    operation="replace", old_text="line3", content="THREE",
                ),
            )

    final = stored["data"].decode()
    assert "Edited" in r1 and "Edited" in r2
    assert "ONE" in final and "THREE" in final


class TestCollectionManagerMultiEdit:
    """The 'edits' batch applies ops sequentially to the evolving text and
    persists exactly once (all-or-nothing)."""

    def _tool_with_file(self, data: bytes):
        helper = TestCollectionManagerTool()
        tool = helper._make_tool()
        stored = {"data": data}

        async def _get(collection, source):
            return {"file_data": stored["data"]}

        async def _save(collection_name, source, filename, content_type, data):
            stored["data"] = data

        stack = helper._patch_collections(tool)
        mock_store = patch("app.stores.RawFileStore")
        return helper, tool, stored, _get, _save, stack, mock_store

    async def _edit(self, fixture, **kwargs):
        helper, tool, stored, _get, _save, stack, mock_store_cm = fixture
        with stack, mock_store_cm as mock_store:
            mock_store.return_value.get = _get
            mock_store.return_value.save = _save
            return await tool.execute(
                "edit", collection="test_collection", filename="f.md", **kwargs
            )

    @pytest.mark.asyncio
    async def test_edits_apply_sequentially(self):
        fx = self._tool_with_file(b"alpha\nbeta\ngamma\n")
        res = await self._edit(fx, edits=[
            {"operation": "replace", "old_text": "beta", "content": "BETA"},
            {"operation": "insert", "line": -1, "content": "delta\n"},
            {"operation": "remove", "text_snippet": "alpha\n"},
        ])
        assert "applied 3 edit(s) (replace, insert, remove)" in res
        assert "4 lines" in res  # trailing \n counts — same convention as read
        assert fx[2]["data"] == b"BETA\ngamma\ndelta\n"

    @pytest.mark.asyncio
    async def test_edits_later_op_sees_earlier_result(self):
        """Line numbers refer to the post-previous-edit state."""
        fx = self._tool_with_file(b"1\n2\n3\n")
        await self._edit(fx, edits=[
            {"operation": "remove", "lines_start": 1, "lines_end": 1},
            # after removing line 1, "2" is now the first line
            {"operation": "insert", "line": 1, "content": "X\n"},
        ])
        assert fx[2]["data"] == b"X\n2\n3\n"

    @pytest.mark.asyncio
    async def test_edits_failure_saves_nothing(self):
        fx = self._tool_with_file(b"a\nb\n")
        with pytest.raises(ValueError, match=r"edit #1.*no changes saved"):
            await self._edit(fx, edits=[
                {"operation": "replace", "old_text": "a", "content": "A"},
                {"operation": "replace", "old_text": "MISSING", "content": "x"},
            ])
        assert fx[2]["data"] == b"a\nb\n"  # first edit rolled back too

    @pytest.mark.asyncio
    async def test_edits_rejects_operation_combo(self):
        fx = self._tool_with_file(b"a\n")
        with pytest.raises(ValueError, match="not both"):
            await self._edit(
                fx, operation="replace", old_text="a", content="b",
                edits=[{"operation": "remove", "text_snippet": "a"}],
            )
        assert fx[2]["data"] == b"a\n"

    @pytest.mark.asyncio
    async def test_edits_rejects_empty_and_oversized(self):
        fx = self._tool_with_file(b"a\n")
        with pytest.raises(ValueError, match="non-empty"):
            await self._edit(fx, edits=[])
        big = [{"operation": "replace", "old_text": "a", "content": "b"}] * 51
        with pytest.raises(ValueError, match="max 50"):
            await self._edit(fx, edits=big)
        assert fx[2]["data"] == b"a\n"

    @pytest.mark.asyncio
    async def test_edits_coerces_stringified_args(self):
        fx = self._tool_with_file(b"1\n2\n")
        res = await self._edit(
            fx, edits=[{"operation": "insert", "line": "2", "content": "X\n"}]
        )
        assert "applied 1 edit(s)" in res
        assert fx[2]["data"] == b"1\nX\n2\n"

    def test_edits_schema_property_exists(self):
        tool = TestCollectionManagerTool()._make_tool()
        props = tool.tool_schema["function"]["parameters"]["properties"]
        assert "edits" in props
        assert props["edits"]["type"] == "array"
