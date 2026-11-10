"""Tests for the Wikipedia search builtin tool."""

import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from agent.tools.builtin_tools.wikipedia_search import WikipediaSearchTool


@pytest.fixture
def tool():
    return WikipediaSearchTool()


@pytest.mark.asyncio
async def test_execute_sends_user_agent(tool):
    """Ensure the User-Agent header is present on the search request."""
    fake_data = json.dumps({
        "query": {"search": [{"title": "Test", "snippet": "", "pageid": 1}]}
    }).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.read.return_value = fake_data
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        result = await tool.execute(query="hello", language="en", limit=1, fetch_extracts=False)

    assert "hello" in result
    req = mock_urlopen.call_args[0][0]
    from agent.tools.builtin_tools.wikipedia_search import _USER_AGENT
    assert req.get_header("User-agent") == _USER_AGENT


def _mock_urlopen(payload: dict):
    """Context-manager-shaped urlopen mock returning *payload* as JSON."""
    fake = json.dumps(payload).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = fake
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _read_payload(title: str, extract: str):
    return {"query": {"pages": {"123": {"pageid": 123, "title": title, "extract": extract}}}}


@pytest.mark.asyncio
async def test_read_returns_full_article(tool):
    extract = "Intro line.\n\n== History ==\nLong ago stuff.\n\n== Details ==\nMore text."
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(_read_payload("Cobot", extract))):
        result = await tool.execute(action="read", title="Cobot")
    assert "Wikipedia article: Cobot (en)" in result
    assert "https://en.wikipedia.org/wiki/Cobot" in result
    assert "== History ==" in result
    assert "== Details ==" in result
    assert "More text." in result
    # line numbers rendered by read_partial_text
    assert "│" in result


@pytest.mark.asyncio
async def test_read_query_fallback_as_title(tool):
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(_read_payload("Cobot", "body"))) as m:
        result = await tool.execute(action="read", query="Cobot")
    assert "Wikipedia article: Cobot" in result
    url = m.call_args[0][0].full_url
    assert "titles=Cobot" in url


@pytest.mark.asyncio
async def test_read_missing_title_errors(tool):
    result = await tool.execute(action="read")
    assert "title is required" in result


@pytest.mark.asyncio
async def test_read_missing_article(tool):
    payload = {"query": {"pages": {"-1": {"ns": 0, "title": "Nope", "missing": ""}}}}
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        result = await tool.execute(action="read", title="Nope")
    assert "not found" in result


@pytest.mark.asyncio
async def test_read_paging_offset(tool):
    extract = "line\n" * 5_000  # ~25k chars
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(_read_payload("Big", extract))):
        result = await tool.execute(action="read", title="Big", offset=10_000)
    # excerpt footer reports the served range; first shown line is ~2001
    assert "excerpt" in result
    assert "2001│" in result
    assert "1│" not in result.split("2001│")[0]


@pytest.mark.asyncio
async def test_read_pattern_filter(tool):
    extract = "alpha intro\n\n== Beta ==\nbeta content\n\ngamma tail"
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(_read_payload("T", extract))):
        result = await tool.execute(action="read", title="T", pattern="beta content")
    assert "beta content" in result


@pytest.mark.asyncio
async def test_read_api_failure_message(tool):
    with patch("urllib.request.urlopen", side_effect=OSError("conn refused")):
        result = await tool.execute(action="read", title="X")
    assert "Read failed" in result


@pytest.mark.asyncio
async def test_unknown_action(tool):
    result = await tool.execute(action="bogus", query="x")
    assert "Unknown action" in result


@pytest.mark.asyncio
async def test_search_requires_query(tool):
    result = await tool.execute(action="search")
    assert "query is required" in result
