"""Unit tests for agent.tools.builtin_tools.common helpers."""

from __future__ import annotations

import re

import pytest

from agent.tools.builtin_tools.common import read_partial_text


class TestOffsetPaging:
    def test_offset_snaps_back_to_line_start(self):
        """An offset inside a line must return that whole line — never skip it."""
        text = "line1\nline2\nline3\nline4\nline5"
        out = read_partial_text(text, offset=8, limit=10, hard_limit=100)
        assert "line2" in out
        assert "line1" not in out
        assert "chars 6-11 of 29" in out
        assert "continue with offset=12" in out
        assert "lines_start=3" in out

    def test_offset_pages_tile_document_without_gaps(self):
        """Following the footer hint must cover every line exactly once."""
        text = "\n".join(f"l{i:02d}" for i in range(1, 51))  # 50 lines of 4 chars
        seen = set()
        offset = 0
        for _ in range(20):
            out = read_partial_text(text, offset=offset, limit=60, hard_limit=100)
            for m in re.finditer(r"l(\d\d)", out):
                seen.add(int(m.group(1)))
            nxt = re.search(r"continue with offset=(\d+)", out)
            if nxt is None:
                break
            offset = int(nxt.group(1))
        assert seen == set(range(1, 51))

    def test_offset_boundary_word_not_split(self):
        """Paging must never cut a word mid-token."""
        text = "alpha beta gamma\ndelta epsilon zeta\neta theta iota"
        out = read_partial_text(text, offset=0, limit=20, hard_limit=100)
        body = out.split("… (")[0]
        # the first line alone is 16 chars; limit 20 snaps at its newline
        assert body.rstrip() == "alpha beta gamma"
        assert "chars 0-16" in out

    def test_offset_inside_first_line_shows_it_whole(self):
        """Offset before the first newline snaps to line 1 start."""
        text = "first line here\nsecond\nthird"
        out = read_partial_text(text, offset=10, limit=20, hard_limit=100)
        assert "first line here" in out
        assert "second" not in out.split("… (")[0]
        assert "chars 0-15" in out

    def test_trailing_newline_counts_as_eof(self):
        """A read ending on the last newline must say end of document."""
        text = "a\nb\nc\n"
        out = read_partial_text(text, offset=2, hard_limit=100)
        assert "end of document" in out
        assert "truncated" not in out

    def test_tail_read_is_excerpt_not_truncated(self):
        text = "a\nb\nc"
        out = read_partial_text(text, offset=3, hard_limit=100)
        assert "excerpt" in out
        assert "end of document" in out
        assert "truncated" not in out

    def test_single_line_doc_keeps_raw_offset(self):
        """Docs without newlines keep the raw offset (mid-line cut noted)."""
        text = "0123456789abcdef"
        out = read_partial_text(text, offset=10, hard_limit=100)
        assert "abcdef" in out
        assert "cut inside long line" in out

    def test_offset_beyond_end(self):
        out = read_partial_text("short", offset=99, hard_limit=100)
        assert "no content" in out
        assert "5 chars" in out

    def test_negative_offset(self):
        text = "0123456789"
        out = read_partial_text(text, offset=-3, hard_limit=100)
        assert "789" in out
        assert "end of document" in out

    def test_offset_inside_giant_first_line_progresses(self):
        """Offset inside a first line longer than the limit must not snap
        back to 0 — that would loop page 1 forever."""
        text = "x" * 500 + "\ntail"
        out = read_partial_text(text, offset=250, limit=200, hard_limit=200)
        body = out.split("… (")[0].rstrip("\n")
        assert body.startswith("x")
        assert len(body) <= 200
        assert "cut inside long line" in out
        nxt = re.search(r"continue with offset=(\d+)", out)
        assert nxt and int(nxt.group(1)) > 250  # keeps making progress


class TestLinePaging:
    def test_char_cap_reports_actual_end_line(self):
        """Truncated line-ranges must report the *shown* end line + next hint."""
        text = "\n".join("x" * 100 for _ in range(10))
        out = read_partial_text(text, lines_start=1, lines_end=10, hard_limit=250)
        assert "truncated" in out
        assert "lines 1-2 of 10" in out
        assert "continue with lines_start=3" in out
        assert "or offset=" in out

    def test_line_numbers_use_real_indices(self):
        text = "alpha\nbeta\ngamma"
        out = read_partial_text(
            text, lines_start=2, lines_end=3, line_numbers=True, hard_limit=100
        )
        assert "2│ beta" in out
        assert "3│ gamma" in out
        assert "1│" not in out

    def test_lines_out_of_range(self):
        out = read_partial_text("a\nb", lines_start=10, lines_end=20, hard_limit=100)
        assert "no content" in out
        assert "10-20" in out

    def test_full_lines_range_no_footer(self):
        out = read_partial_text("a\nb\nc", lines_start=1, lines_end=3, hard_limit=100)
        assert out == "a\nb\nc"

    def test_giant_single_line_respects_limit_and_offers_offset(self):
        """A line bigger than the cap must be cut, not returned whole."""
        text = "y" * 800 + "\nlast"
        out = read_partial_text(text, lines_start=1, lines_end=2, hard_limit=300)
        body = out.split("… (")[0].rstrip("\n")
        assert len(body) <= 300
        assert "truncated" in out
        assert "offset=" in out  # offset-hint continues inside the same line
        nxt = re.search(r"offset=(\d+)", out)
        assert nxt and 0 < int(nxt.group(1)) < 800


class TestPatternMode:
    def test_pattern_returns_context_windows(self):
        lines = [f"line{i}" for i in range(1, 21)]
        lines[7] = "hit-one"
        lines[15] = "hit-two"
        text = "\n".join(lines)
        out = read_partial_text(text, pattern="hit", context_lines=1, hard_limit=500)
        assert "8│ hit-one" in out
        assert "7│ line7" in out  # context before
        assert "9│ line9" in out  # context after
        assert "16│ hit-two" in out
        assert "⋮" in out  # window separator
        assert "2 matches" in out
        assert "1│ line1" not in out

    def test_pattern_windows_merge(self):
        lines = [f"l{i}" for i in range(10)]
        lines[2] = "hit-a"
        lines[3] = "hit-b"
        text = "\n".join(lines)
        out = read_partial_text(text, pattern="hit", context_lines=2, hard_limit=500)
        assert "1 window" in out

    def test_pattern_case_insensitive(self):
        out = read_partial_text("Hello WORLD", pattern="world", hard_limit=100)
        assert "WORLD" in out

    def test_pattern_no_match(self):
        out = read_partial_text("a\nb\nc", pattern="zzz", hard_limit=100)
        assert "no matches" in out
        assert "'zzz'" in out

    def test_pattern_cap_reports_cutoff(self):
        """More windows than fit must be reported, not silently dropped."""
        lines = []
        for i in range(30):
            lines.extend(
                [
                    f"pad-{i}-a long padding line to grow output",
                    f"pad-{i}-b filler",
                    f"match here {i}",
                    f"pad-{i}-c filler",
                    f"pad-{i}-d filler",
                ]
            )
        text = "\n".join(lines)
        out = read_partial_text(
            text, pattern="match here", context_lines=1, hard_limit=500
        )
        assert "cut off" in out or "truncated" in out
        assert "matches" in out
        assert len(out) < 800


class TestLineNumbers:
    def test_numbered_output_in_offset_mode(self):
        text = "first\nsecond\nthird"
        out = read_partial_text(text, offset=6, line_numbers=True, hard_limit=100)
        assert "2│ second" in out
        assert "3│ third" in out

    def test_full_document_numbered(self):
        out = read_partial_text("a\nb", line_numbers=True, hard_limit=100)
        assert "1│ a" in out
        assert "2│ b" in out
