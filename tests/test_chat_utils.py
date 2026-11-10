from __future__ import annotations

import pytest

from app.chat_utils import (
    build_message_metrics,
    extract_reasoning_text,
    extract_usage_metadata,
    finalize_token_fields,
    render_system_instructions,
)


def test_extract_usage_metadata_from_dict_response_metadata():
    result = {
        "response_metadata": {
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "reasoning_tokens": 5,
            }
        }
    }
    meta = extract_usage_metadata(result)
    assert meta == {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "reasoning_tokens": 5,
        "cache_tokens": None,
    }


def test_extract_usage_metadata_from_dict_usage_metadata():
    result = {
        "usage_metadata": {
            "input_tokens": 8,
            "output_tokens": 16,
            "total_tokens": 24,
        }
    }
    meta = extract_usage_metadata(result)
    assert meta == {
        "prompt_tokens": 8,
        "completion_tokens": 16,
        "total_tokens": 24,
        "reasoning_tokens": None,
        "cache_tokens": None,
    }


def test_extract_usage_metadata_anthropic_cache_tokens():
    """Anthropic-style input_token_details carry cache tokens outside
    input_tokens — they must be captured so run totals can include them."""
    result = {
        "usage_metadata": {
            "input_tokens": 8,
            "output_tokens": 16,
            "total_tokens": 24,
            "input_token_details": {
                "cache_read": 500,
                "cache_creation": 120,
            },
        }
    }
    meta = extract_usage_metadata(result)
    assert meta == {
        "prompt_tokens": 8,
        "completion_tokens": 16,
        "total_tokens": 24,
        "reasoning_tokens": None,
        "cache_tokens": 620,
    }


def test_extract_usage_metadata_openai_cached_tokens_not_double_counted():
    """OpenAI's prompt_tokens_details.cached_tokens is already inside
    prompt_tokens — it must NOT surface as cache_tokens."""
    result = {
        "usage_metadata": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {
                "cached_tokens": 60,
            },
        }
    }
    meta = extract_usage_metadata(result)
    assert meta == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "reasoning_tokens": None,
        "cache_tokens": None,
    }


def test_extract_usage_metadata_from_completion_tokens_details():
    result = {
        "usage_metadata": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "completion_tokens_details": {
                "reasoning_tokens": 7,
            },
        }
    }
    meta = extract_usage_metadata(result)
    assert meta == {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "reasoning_tokens": 7,
        "cache_tokens": None,
    }


def test_extract_usage_metadata_from_output_token_details():
    result = {
        "usage_metadata": {
            "prompt_tokens": 12,
            "completion_tokens": 28,
            "total_tokens": 40,
            "output_token_details": {
                "reasoning": 9,
            },
        }
    }
    meta = extract_usage_metadata(result)
    assert meta == {
        "prompt_tokens": 12,
        "completion_tokens": 28,
        "total_tokens": 40,
        "reasoning_tokens": 9,
        "cache_tokens": None,
    }


def test_extract_usage_metadata_returns_none_when_missing():
    assert extract_usage_metadata({}) is None
    assert extract_usage_metadata(None) is None
    assert extract_usage_metadata({"answer": "hello"}) is None


def test_extract_usage_metadata_from_object_with_attrs():
    class FakeMessage:
        response_metadata = {
            "token_usage": {
                "prompt_tokens": 5,
                "completion_tokens": 10,
                "total_tokens": 15,
            }
        }
        usage_metadata = None

    meta = extract_usage_metadata(FakeMessage())
    assert meta == {
        "prompt_tokens": 5,
        "completion_tokens": 10,
        "total_tokens": 15,
        "reasoning_tokens": None,
        "cache_tokens": None,
    }


def test_finalize_token_fields_includes_cache_in_total():
    fields = finalize_token_fields(prompt=100, completion=50, total=150, cache=40)
    assert fields["total_tokens"] == 190
    assert fields["cache_tokens"] == 40
    assert fields["answer_tokens"] == 50


def test_finalize_token_fields_cache_none_keeps_total():
    fields = finalize_token_fields(prompt=100, completion=50, total=150)
    assert fields["total_tokens"] == 150
    assert fields["cache_tokens"] is None


def test_build_message_metrics_full_payload():
    info = {
        "_llm_usage_metadata": {
            "prompt_tokens": 1000,
            "completion_tokens": 200,
            "total_tokens": 1200,
            "reasoning_tokens": 60,
            "iterations": 3,
            "tool_result_tokens": 300,
            "cache_tokens": 50,
            "observer_tokens": 90,
            "observer_calls": 2,
            "compression_tokens": 120,
            "subagent_tokens": 400,
            "per_call": [{"p": 500, "c": 100, "r": 30, "ms": 800.0},
                         {"p": 500, "c": 100, "r": 30, "ms": 700.0}],
        },
        "_final_answer_usage": {
            "prompt_tokens": 800,
            "completion_tokens": 100,
            "reasoning_tokens": 20,
            "_response_time_ms": 1500.0,
        },
        "rag_context_tokens": 250,
    }
    m = build_message_metrics(
        retrieval_info=info,
        assistant_text="the answer",
        reasoning_text=None,
        response_ms=5000.0,
        stream_start=1000.0,
        first_token_at=1000.4,
        first_answer_token_at=1001.2,
        subagent_count=2,
    )
    # provider total (1200) lacked the 50 cache tokens → lifted to the true sum
    assert m["total_tokens"] == 1250
    assert m["answer_tokens"] == 140
    assert m["cache_tokens"] == 50
    assert m["observer_tokens"] == 90
    assert m["compression_tokens"] == 120
    assert m["subagent_tokens"] == 400
    assert m["final_prompt_tokens"] == 800
    assert m["final_completion_tokens"] == 100
    assert m["final_answer_ms"] == 1500.0
    assert m["time_to_first_token_ms"] == 400.0
    assert m["time_to_first_answer_token_ms"] == 1200.0
    assert m["tokens_per_second"] == 66.67
    assert m["agent_iterations"] == 3
    # 3 iterations + 2 observer checks — observer calls are real LLM calls
    # and the live chip already counts them, so the badge must match.
    assert m["llm_calls"] == 5
    assert m["subagent_count"] == 2
    assert m["rag_context_tokens"] == 250
    assert m["per_call"] == [{"p": 500, "c": 100, "r": 30, "ms": 800.0},
                             {"p": 500, "c": 100, "r": 30, "ms": 700.0}]


def test_build_message_metrics_estimate_fallback():
    m = build_message_metrics(
        retrieval_info={},
        assistant_text="short",
        reasoning_text=None,
        response_ms=100.0,
        prompt_estimate=42,
    )
    assert m["prompt_tokens"] == 42
    assert m["time_to_first_token_ms"] is None
    assert m["response_time_ms"] == 100.0


def test_extract_reasoning_text_from_dict_additional_kwargs():
    result = {
        "additional_kwargs": {
            "reasoning_content": "Thinking about the problem...",
        }
    }
    assert extract_reasoning_text(result) == "Thinking about the problem..."


def test_extract_reasoning_text_from_dict_reasoning_key():
    result = {
        "additional_kwargs": {
            "reasoning": "Alternative reasoning key",
        }
    }
    assert extract_reasoning_text(result) == "Alternative reasoning key"


def test_extract_reasoning_text_from_dict_direct_reasoning_key():
    result = {"reasoning": "Direct dict reasoning"}
    assert extract_reasoning_text(result) == "Direct dict reasoning"


def test_extract_reasoning_text_from_response_metadata_message():
    result = {
        "response_metadata": {
            "message": {
                "reasoning_content": "Reasoning in message metadata",
            }
        }
    }
    assert extract_reasoning_text(result) == "Reasoning in message metadata"


def test_extract_reasoning_text_from_object_attributes():
    class FakeChunk:
        additional_kwargs = {"reasoning_content": "Object attribute reasoning"}

    assert extract_reasoning_text(FakeChunk()) == "Object attribute reasoning"


def test_extract_reasoning_text_from_chat_generation_chunk():
    class FakeMessage:
        additional_kwargs = {"reasoning_content": "Reasoning in message.additional_kwargs"}

    class FakeGenerationChunk:
        message = FakeMessage()

    assert extract_reasoning_text(FakeGenerationChunk()) == "Reasoning in message.additional_kwargs"


def test_extract_reasoning_text_from_response_metadata_direct():
    result = {
        "response_metadata": {
            "reasoning_content": "Direct reasoning in response_metadata",
        }
    }
    assert extract_reasoning_text(result) == "Direct reasoning in response_metadata"


def test_extract_reasoning_text_from_response_metadata_delta():
    result = {
        "response_metadata": {
            "delta": {
                "reasoning_content": "Delta reasoning chunk",
            }
        }
    }
    assert extract_reasoning_text(result) == "Delta reasoning chunk"


def test_extract_reasoning_text_returns_empty_when_missing():
    assert extract_reasoning_text({}) == ""
    assert extract_reasoning_text(None) == ""
    assert extract_reasoning_text({"content": "hello"}) == ""


def test_render_system_instructions_replaces_datetime_placeholder():
    text = "Heute ist {{current_datetime}}. Bereit."
    result = render_system_instructions(text)
    assert "{{current_datetime}}" not in result
    # German long-form pattern: Weekday, DD. Month YYYY, HH:MM Uhr
    import re
    assert re.search(r"[A-Za-zäöüÄÖÜ]+, \d{1,2}\. [A-Za-zäöüÄÖÜ]+ \d{4}, \d{2}:\d{2} Uhr", result)


def test_render_system_instructions_replaces_context_placeholder():
    text = "Context: {{context}}"
    result = render_system_instructions(text, rag_context="Doc content")
    assert result == "Context: Doc content"
    assert "{{context}}" not in result


def test_render_system_instructions_no_context_placeholder_unchanged():
    text = "Du bist ein einfacher Assistent."
    assert render_system_instructions(text) == text
    assert render_system_instructions(text, rag_context="Doc") == text


def test_render_system_instructions_empty_string():
    assert render_system_instructions("") == ""
    assert render_system_instructions(None) == ""


def test_render_system_instructions_replaces_agent_memory_placeholder():
    text = "Memory: {{agent_memory}}. Done."
    result = render_system_instructions(text, agent_memory="user likes tea")
    assert result == "Memory: user likes tea. Done."
    assert "{{agent_memory}}" not in result


def test_render_system_instructions_agent_memory_placeholder_none():
    text = "Memory: {{agent_memory}}. Done."
    result = render_system_instructions(text)
    assert result == "Memory: . Done."
    assert "{{agent_memory}}" not in result


def test_render_system_instructions_no_agent_memory_placeholder_unchanged():
    text = "Du bist ein einfacher Assistent."
    assert render_system_instructions(text, agent_memory="foo") == text


def test_build_message_metrics_estimated_flag_and_tps_precedence():
    """Aggregate tokens_per_second (output/LLM-time) wins over the final
    answer's rate; estimated_iterations surfaces as the 'estimated' flag."""
    info = {
        "_llm_usage_metadata": {
            "prompt_tokens": 1000,
            "completion_tokens": 200,
            "total_tokens": 1200,
            "iterations": 2,
            "tokens_per_second": 21.5,
            "estimated_iterations": 2,
        },
        "_final_answer_usage": {
            "prompt_tokens": 800,
            "completion_tokens": 100,
            "_response_time_ms": 1500.0,
        },
    }
    m = build_message_metrics(
        retrieval_info=info,
        assistant_text="answer",
        reasoning_text=None,
        response_ms=5000.0,
    )
    assert m["tokens_per_second"] == 21.5  # aggregate wins over 100/1.5=66.67
    assert m["estimated"] is True


def test_build_message_metrics_tps_falls_back_to_final_answer():
    """Without an aggregate rate the final answer's call rate is used."""
    info = {
        "_llm_usage_metadata": {
            "prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200,
        },
        "_final_answer_usage": {
            "completion_tokens": 100, "_response_time_ms": 1500.0,
        },
    }
    m = build_message_metrics(
        retrieval_info=info, assistant_text="answer",
        reasoning_text=None, response_ms=5000.0)
    assert m["tokens_per_second"] == 66.67
    assert "estimated" not in m
