"""Unit tests for the metrics token normalisation + timing markers.

Covers ``finalize_token_fields`` (the single place that enforces the
``reasoning <= completion <= total`` invariants), ``collect_tool_names``,
and the ``ActiveChatRun.record_token`` marker semantics
(first model token of ANY kind vs. first visible answer token).
"""
from __future__ import annotations

import pytest

from app.chat_runs import ActiveChatRun
from app.chat_utils import collect_tool_names, finalize_token_fields


class TestFinalizeTokenFields:
    def test_provider_shape_reasoning_inside_completion(self):
        # Provider convention: completion_tokens already contains reasoning.
        fields = finalize_token_fields(
            prompt=100, completion=50, total=150, reasoning=20
        )
        assert fields["prompt_tokens"] == 100
        assert fields["completion_tokens"] == 50
        assert fields["reasoning_tokens"] == 20
        assert fields["answer_tokens"] == 30
        assert fields["total_tokens"] == 150

    def test_estimated_path_maps_answer_plus_reasoning_to_completion(self):
        # Estimation counts only visible answer text -> lifted onto the
        # provider convention so completion = answer + reasoning.
        fields = finalize_token_fields(
            prompt=200, est_answer=40, est_reasoning=15
        )
        assert fields["completion_tokens"] == 55
        assert fields["reasoning_tokens"] == 15
        assert fields["answer_tokens"] == 40
        assert fields["total_tokens"] == 255

    def test_missing_total_is_derived(self):
        fields = finalize_token_fields(prompt=10, completion=5, reasoning=2)
        assert fields["total_tokens"] == 15

    def test_undercounted_total_is_clamped_up(self):
        # Provider total that misses a second call gets corrected.
        fields = finalize_token_fields(
            prompt=100, completion=50, total=140, reasoning=10
        )
        assert fields["total_tokens"] == 150

    def test_reasoning_never_exceeds_completion(self):
        fields = finalize_token_fields(
            prompt=10, completion=5, total=15, reasoning=9
        )
        assert fields["reasoning_tokens"] == 5
        assert fields["answer_tokens"] == 0

    def test_no_reasoning(self):
        fields = finalize_token_fields(prompt=10, completion=5, total=15)
        assert fields["reasoning_tokens"] is None
        assert fields["answer_tokens"] == 5

    def test_all_none_stays_none(self):
        fields = finalize_token_fields()
        assert fields == {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "reasoning_tokens": None,
            "answer_tokens": None,
            "cache_tokens": None,
        }

    def test_cache_lifts_total(self):
        fields = finalize_token_fields(prompt=10, completion=5, total=15, cache=30)
        assert fields["total_tokens"] == 45
        assert fields["cache_tokens"] == 30

    def test_invariants_hold(self):
        for kwargs in (
            {"prompt": 10, "completion": 5, "total": 15, "reasoning": 2},
            {"prompt": 0, "est_answer": 3},
            {"prompt": 100, "completion": 200, "total": 100, "reasoning": 500},
            {"completion": 42, "total": 42},
        ):
            f = finalize_token_fields(**kwargs)
            if f["reasoning_tokens"] is not None and f["completion_tokens"] is not None:
                assert f["reasoning_tokens"] <= f["completion_tokens"]
            if f["completion_tokens"] is not None and f["total_tokens"] is not None:
                assert f["completion_tokens"] <= f["total_tokens"]
            if f["answer_tokens"] is not None and f["completion_tokens"] is not None:
                assert f["answer_tokens"] + (f["reasoning_tokens"] or 0) == f["completion_tokens"]


class TestCollectToolNames:
    def test_normalized_names(self):
        calls = [{"name": "rag_search"}, {"name": "web_search"}, {"name": "rag_search"}]
        assert collect_tool_names(calls) == ["rag_search", "web_search"]

    def test_agent_trace_names(self):
        calls = [{"tool_name": "wait"}, {"tool_name": "subagent_spawn"}]
        assert collect_tool_names(calls) == ["subagent_spawn", "wait"]

    def test_mixed_and_dirty_entries(self):
        calls = [{"name": "a"}, {"tool_name": "b"}, "junk", {"args": {}}, None]
        assert collect_tool_names(calls) == ["a", "b"]

    def test_empty(self):
        assert collect_tool_names(None) is None
        assert collect_tool_names([]) is None
        assert collect_tool_names([{"args": {}}]) is None


def _make_run() -> ActiveChatRun:
    return ActiveChatRun(
        conversation_id="conv-1",
        model_name="test-model",
        user_message="hello",
    )


class TestRecordTokenMarkers:
    def test_reasoning_first_sets_any_but_not_answer(self):
        run = _make_run()
        run.record_token({"reasoning": "thinking..."})
        assert run.first_token_at is not None
        assert run.first_reasoning_token_at is not None
        assert run.first_answer_token_at is None

    def test_content_first_sets_both(self):
        run = _make_run()
        run.record_token({"content": "Hello"})
        assert run.first_token_at is not None
        assert run.first_answer_token_at is not None
        assert run.first_reasoning_token_at is None

    def test_reasoning_then_content_order(self):
        run = _make_run()
        run.record_token({"reasoning": "r1"})
        first = run.first_token_at
        run.record_token({"content": "a"})
        assert run.first_token_at == first  # stays at the reasoning marker
        assert run.first_answer_token_at is not None
        assert run.first_answer_token_at >= run.first_token_at

    def test_tool_only_tokens_do_not_mark(self):
        run = _make_run()
        run.record_token({"tool_progress": {"id": "t1", "status": "running"}})
        run.record_token({"subagent": {"id": "s1"}})
        assert run.produced_any is True
        assert run.first_token_at is None
        assert run.first_answer_token_at is None
        assert run.first_reasoning_token_at is None

    def test_first_marker_not_overwritten(self):
        run = _make_run()
        run.record_token({"content": "a"})
        first = run.first_token_at
        first_answer = run.first_answer_token_at
        run.record_token({"content": "b", "reasoning": "r"})
        assert run.first_token_at == first
        assert run.first_answer_token_at == first_answer


class TestTriggerTimeoutBounds:
    def test_defaults(self):
        from app.routers.triggers import TriggerCreatePayload, TriggerUpdatePayload

        payload = TriggerCreatePayload(
            name="t", assistant_name="a", prompt="p", schedule_type="interval"
        )
        assert payload.timeout_seconds == 600
        update = TriggerUpdatePayload(
            name="t", assistant_name="a", prompt="p", schedule_type="interval"
        )
        assert update.timeout_seconds == 600

    def test_bounds(self):
        from pydantic import ValidationError

        from app.routers.triggers import TriggerCreatePayload

        with pytest.raises(ValidationError):
            TriggerCreatePayload(
                name="t", assistant_name="a", prompt="p",
                schedule_type="interval", timeout_seconds=5,
            )
        with pytest.raises(ValidationError):
            TriggerCreatePayload(
                name="t", assistant_name="a", prompt="p",
                schedule_type="interval", timeout_seconds=3601,
            )
        ok = TriggerCreatePayload(
            name="t", assistant_name="a", prompt="p",
            schedule_type="interval", timeout_seconds=3600,
        )
        assert ok.timeout_seconds == 3600
