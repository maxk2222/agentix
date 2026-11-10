"""Tests for the Bing HTML parser — sponsored results must be filtered."""

from __future__ import annotations

from agent.tools.builtin_tools.common import parse_bing_results


def _li(body: str) -> str:
    return f'<li class="b_algo">{body}</li>'


def _result(url: str, title: str = "T", snippet: str = "S") -> str:
    return f'<h2><a href="{url}">{title}</a></h2><p>{snippet}</p>'


class TestParseBingResults:
    def test_organic_results_parsed(self):
        html = _li(_result("https://a.example.com/", "Alpha", "snip a"))
        results = parse_bing_results(html)
        assert len(results) == 1
        assert results[0]["url"] == "https://a.example.com/"
        assert results[0]["title"] == "Alpha"

    def test_ad_segments_filtered(self):
        html = (
            _li(_result("https://ad1.example.com/buy") + '<span class="b_adlabel">Ad</span>')
            + _li(_result("https://real.example.com/", "Real", "organic"))
            + _li(_result("https://ad2.example.com/shop") + '<div class="b_adTop"></div>')
        )
        results = parse_bing_results(html)
        urls = [r["url"] for r in results]
        assert urls == ["https://real.example.com/"]

    def test_adlabel_variants_filtered(self):
        for marker in ('class="b_adlabel"', 'class="b_adSlug"', 'class="b_adBottom"'):
            html = _li(_result("https://sponsor.example.com/") + f'<span {marker}>x</span>')
            assert parse_bing_results(html) == []

    def test_max_results_respected(self):
        html = "".join(
            _li(_result(f"https://site{i}.example.com/")) for i in range(10)
        )
        results = parse_bing_results(html, max_results=3)
        assert len(results) == 3

    def test_duplicate_urls_deduped(self):
        html = _li(_result("https://dup.example.com/")) + _li(
            _result("https://dup.example.com/")
        )
        assert len(parse_bing_results(html)) == 1
