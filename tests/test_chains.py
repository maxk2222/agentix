"""Tests for agent.chains.ChainBuilder."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from agent.chains import ChainBuilder
from agent.config import AppConfig


class _FakePromptTemplate:
    """Minimal stand-in for ChatPromptTemplate that supports the pipe operator."""

    def __init__(self, messages: list) -> None:
        self.messages = messages

    def partial(self, **kwargs: Any) -> "_FakePromptTemplate":
        return self

    def __or__(self, other: Any) -> "_FakePromptTemplate":
        return self


def _capture_system_texts(captured: list):
    """Capture system text from both ("system", str) tuples and SystemMessage
    objects passed to ChatPromptTemplate.from_messages."""

    def _capture(messages):
        for msg in messages:
            if isinstance(msg, tuple) and msg[0] == "system":
                captured.append(msg[1])
            elif isinstance(msg, SystemMessage):
                captured.append(str(msg.content))
        return _FakePromptTemplate(messages)

    return _capture


@pytest.mark.asyncio
async def test_build_chains_appends_context_when_no_placeholder(tmp_path):
    """If system_instructions lacks {{context}} and rag_context is provided, it should be auto-appended."""
    instructions = {
        "TestAssistant": {
            "model": "gpt-4",
            "temperature": 0.5,
            "system_instructions": "You are helpful.",
        }
    }
    config = AppConfig(
        data_dir=tmp_path / "data",
        instructions_file=tmp_path / "instructions.yaml",
        pg_host="localhost",
        pg_port=5432,
        pg_database="test",
        pg_user="test",
        pg_password="test",
    )
    builder = ChainBuilder(config=config, instructions=instructions)

    captured_prompts: list[str] = []

    with patch("agent.chains.ChatPromptTemplate.from_messages", side_effect=_capture_system_texts(captured_prompts)):
        with patch.object(
            builder, "_get_llm_for_assistant", new_callable=AsyncMock, return_value=object()
        ):
            await builder.build_chains(
                "TestAssistant",
                use_contextualize=False,
                retriever=None,
                retrieval_degraded=True,
                rag_context="Doc content",
            )

    assert len(captured_prompts) >= 1
    system_prompt = next(p for p in captured_prompts if p != "{context}")
    assert "You are helpful." in system_prompt
    assert "Relevant context:" in system_prompt
    assert "Doc content" in system_prompt
    assert "{{context}}" not in system_prompt


@pytest.mark.asyncio
async def test_build_chains_replaces_context_placeholder(tmp_path):
    """If system_instructions contains {{context}}, it should be replaced with rag_context."""
    instructions = {
        "TestAssistant": {
            "model": "gpt-4",
            "temperature": 0.5,
            "system_instructions": "Context: {{context}}\n\nBe helpful.",
        }
    }
    config = AppConfig(
        data_dir=tmp_path / "data",
        instructions_file=tmp_path / "instructions.yaml",
        pg_host="localhost",
        pg_port=5432,
        pg_database="test",
        pg_user="test",
        pg_password="test",
    )
    builder = ChainBuilder(config=config, instructions=instructions)

    captured_prompts: list[str] = []

    with patch("agent.chains.ChatPromptTemplate.from_messages", side_effect=_capture_system_texts(captured_prompts)):
        with patch.object(
            builder, "_get_llm_for_assistant", new_callable=AsyncMock, return_value=object()
        ):
            await builder.build_chains(
                "TestAssistant",
                use_contextualize=False,
                retriever=None,
                retrieval_degraded=True,
                rag_context="Doc content",
            )

    assert len(captured_prompts) >= 1
    system_prompt = next(p for p in captured_prompts if p != "{context}")
    assert "Context: Doc content" in system_prompt
    assert "Be helpful." in system_prompt
    assert "{{context}}" not in system_prompt


@pytest.mark.asyncio
async def test_build_chains_appends_emotion_hint(tmp_path):
    """If emotion_hint is provided, it should be appended to the system prompt."""
    instructions = {
        "TestAssistant": {
            "model": "gpt-4",
            "temperature": 0.5,
            "system_instructions": "You are helpful.",
        }
    }
    config = AppConfig(
        data_dir=tmp_path / "data",
        instructions_file=tmp_path / "instructions.yaml",
        pg_host="localhost",
        pg_port=5432,
        pg_database="test",
        pg_user="test",
        pg_password="test",
    )
    builder = ChainBuilder(config=config, instructions=instructions)

    captured_prompts: list[str] = []

    with patch("agent.chains.ChatPromptTemplate.from_messages", side_effect=_capture_system_texts(captured_prompts)):
        with patch.object(
            builder, "_get_llm_for_assistant", new_callable=AsyncMock, return_value=object()
        ):
            await builder.build_chains(
                "TestAssistant",
                use_contextualize=False,
                retriever=None,
                retrieval_degraded=True,
                rag_context="",
                emotion_hint="[Emotion Context]\nFused: frustrated",
            )

    assert len(captured_prompts) >= 1
    system_prompt = captured_prompts[0]
    assert "You are helpful." in system_prompt
    assert "[Emotion Context]" in system_prompt
    assert "Fused: frustrated" in system_prompt


@pytest.mark.asyncio
async def test_extra_body_passed_to_llm(tmp_path):
    instructions = {
        "TestAssistant": {
            "model": "gpt-4",
            "temperature": 0.5,
            "system_instructions": "You are helpful.",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
    }
    config = AppConfig(
        data_dir=tmp_path / "data",
        instructions_file=tmp_path / "instructions.yaml",
        pg_host="localhost",
        pg_port=5432,
        pg_database="test",
        pg_user="test",
        pg_password="test",
    )
    builder = ChainBuilder(config=config, instructions=instructions)

    with patch("agent.chains.ChatOpenAIWithReasoning") as mock_llm_class:
        mock_instance = mock_llm_class.return_value
        await builder._get_llm_for_assistant("TestAssistant", streaming=False)
        call_kwargs = mock_llm_class.call_args.kwargs
        assert call_kwargs["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}
        }


@pytest.mark.asyncio
async def test_extra_body_merged_with_reasoning(tmp_path):
    instructions = {
        "TestAssistant": {
            "model": "gpt-4",
            "temperature": 0.5,
            "system_instructions": "You are helpful.",
            "reasoning": {"effort": "low"},
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
    }
    config = AppConfig(
        data_dir=tmp_path / "data",
        instructions_file=tmp_path / "instructions.yaml",
        pg_host="localhost",
        pg_port=5432,
        pg_database="test",
        pg_user="test",
        pg_password="test",
    )
    builder = ChainBuilder(config=config, instructions=instructions)

    with patch("agent.chains.ChatOpenAIWithReasoning") as mock_llm_class:
        mock_instance = mock_llm_class.return_value
        await builder._get_llm_for_assistant("TestAssistant", streaming=False)
        call_kwargs = mock_llm_class.call_args.kwargs
        assert call_kwargs["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
        assert call_kwargs["extra_body"]["reasoning"] == {"effort": "low"}


def _make_config(tmp_path) -> AppConfig:
    return AppConfig(
        data_dir=tmp_path / "data",
        instructions_file=tmp_path / "instructions.yaml",
        pg_host="localhost",
        pg_port=5432,
        pg_database="test",
        pg_user="test",
        pg_password="test",
    )


@pytest.mark.asyncio
async def test_enable_thinking_merged_into_extra_body(tmp_path):
    instructions = {
        "TestAssistant": {
            "model": "gpt-4",
            "temperature": 0.5,
            "system_instructions": "You are helpful.",
            "enable_thinking": False,
        }
    }
    builder = ChainBuilder(config=_make_config(tmp_path), instructions=instructions)

    with patch("agent.chains.ChatOpenAIWithReasoning") as mock_llm_class:
        await builder._get_llm_for_assistant("TestAssistant", streaming=False)
        call_kwargs = mock_llm_class.call_args.kwargs
        assert call_kwargs["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}
        }


@pytest.mark.asyncio
async def test_enable_thinking_field_overrides_manual_extra_body(tmp_path):
    """The explicit enable_thinking field wins over the same key in extra_body."""
    instructions = {
        "TestAssistant": {
            "model": "gpt-4",
            "temperature": 0.5,
            "system_instructions": "You are helpful.",
            "enable_thinking": False,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": True},
                "other_key": 1,
            },
        }
    }
    builder = ChainBuilder(config=_make_config(tmp_path), instructions=instructions)

    with patch("agent.chains.ChatOpenAIWithReasoning") as mock_llm_class:
        await builder._get_llm_for_assistant("TestAssistant", streaming=False)
        call_kwargs = mock_llm_class.call_args.kwargs
        assert call_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
        assert call_kwargs["extra_body"]["other_key"] == 1
    # Source dict must not be mutated by the merge
    assert instructions["TestAssistant"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


@pytest.mark.asyncio
async def test_extra_parameters_become_model_kwargs(tmp_path):
    instructions = {
        "TestAssistant": {
            "model": "gpt-4",
            "temperature": 0.5,
            "system_instructions": "You are helpful.",
            "extra_parameters": {"top_p": 0.9, "presence_penalty": 0.3},
        }
    }
    builder = ChainBuilder(config=_make_config(tmp_path), instructions=instructions)

    with patch("agent.chains.ChatOpenAIWithReasoning") as mock_llm_class:
        await builder._get_llm_for_assistant("TestAssistant", streaming=False)
        call_kwargs = mock_llm_class.call_args.kwargs
        assert call_kwargs["model_kwargs"] == {"top_p": 0.9, "presence_penalty": 0.3}


@pytest.mark.asyncio
async def test_extra_parameters_preserve_stream_options_when_streaming(tmp_path):
    instructions = {
        "TestAssistant": {
            "model": "gpt-4",
            "temperature": 0.5,
            "system_instructions": "You are helpful.",
            "extra_parameters": {"top_p": 0.9},
        }
    }
    builder = ChainBuilder(config=_make_config(tmp_path), instructions=instructions)

    with patch("agent.chains.ChatOpenAIWithReasoning") as mock_llm_class:
        await builder._get_llm_for_assistant("TestAssistant", streaming=True)
        call_kwargs = mock_llm_class.call_args.kwargs
        assert call_kwargs["model_kwargs"]["top_p"] == 0.9
        assert call_kwargs["model_kwargs"]["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_agent_llm_receives_extra_body_and_parameters(tmp_path):
    instructions = {
        "TestAssistant": {
            "model": "gpt-4",
            "temperature": 0.5,
            "system_instructions": "You are helpful.",
            "enable_thinking": False,
            "extra_body": {"custom_key": "v"},
            "extra_parameters": {"top_p": 0.8},
        }
    }
    builder = ChainBuilder(config=_make_config(tmp_path), instructions=instructions)

    with patch("agent.chains.ChatOpenAIAgentLLM") as mock_agent_llm:
        await builder._get_agent_llm("TestAssistant")
        call_kwargs = mock_agent_llm.call_args.kwargs
        assert call_kwargs["extra_body"]["custom_key"] == "v"
        assert call_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
        assert call_kwargs["model_kwargs"] == {"top_p": 0.8}


@pytest.mark.asyncio
async def test_clear_cache_clears_agent_llm_cache(tmp_path):
    instructions = {
        "TestAssistant": {
            "model": "gpt-4",
            "temperature": 0.5,
            "system_instructions": "You are helpful.",
        }
    }
    builder = ChainBuilder(config=_make_config(tmp_path), instructions=instructions)

    with patch("agent.chains.ChatOpenAIWithReasoning"), patch("agent.chains.ChatOpenAIAgentLLM"):
        await builder._get_llm_for_assistant("TestAssistant", streaming=False)
        await builder._get_agent_llm("TestAssistant")
        assert builder._llm_cache
        assert builder._agent_llm_cache
        await builder.clear_cache()
        assert not builder._llm_cache
        assert not builder._agent_llm_cache


@pytest.mark.asyncio
async def test_provider_id_separates_llm_cache(tmp_path):
    """Assistants with the same model but different provider_id must not share an LLM."""
    instructions = {
        "AssistantA": {
            "model": "gpt-4",
            "temperature": 0.5,
            "system_instructions": "A.",
            "provider_id": "prov-a",
        },
        "AssistantB": {
            "model": "gpt-4",
            "temperature": 0.5,
            "system_instructions": "B.",
            "provider_id": "prov-b",
        },
    }
    builder = ChainBuilder(config=_make_config(tmp_path), instructions=instructions)

    with patch("agent.chains.ChatOpenAIWithReasoning"):
        await builder._get_llm_for_assistant("AssistantA", streaming=False)
        await builder._get_llm_for_assistant("AssistantB", streaming=False)
        assert len(builder._llm_cache) == 2


@pytest.mark.asyncio
async def test_reasoning_translated_for_qwen_model(tmp_path):
    """Qwen models receive chat_template_kwargs, not a verbatim reasoning key."""
    instructions = {
        "TestAssistant": {
            "model": "Qwen3-8B",
            "temperature": 0.5,
            "system_instructions": "You are helpful.",
            "reasoning": {"enabled": False, "budget": 2048},
        }
    }
    builder = ChainBuilder(config=_make_config(tmp_path), instructions=instructions)

    with patch("agent.chains.ChatOpenAIWithReasoning") as mock_llm_class:
        await builder._get_llm_for_assistant("TestAssistant", streaming=False)
        call_kwargs = mock_llm_class.call_args.kwargs
        assert call_kwargs["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False, "thinking_budget": 2048}
        }


@pytest.mark.asyncio
async def test_reasoning_deep_merges_with_extra_body_for_qwen(tmp_path):
    instructions = {
        "TestAssistant": {
            "model": "Qwen3-8B",
            "temperature": 0.5,
            "system_instructions": "You are helpful.",
            "reasoning": {"enabled": True},
            "extra_body": {"chat_template_kwargs": {"thinking_budget": 512}},
        }
    }
    builder = ChainBuilder(config=_make_config(tmp_path), instructions=instructions)

    with patch("agent.chains.ChatOpenAIWithReasoning") as mock_llm_class:
        await builder._get_llm_for_assistant("TestAssistant", streaming=False)
        call_kwargs = mock_llm_class.call_args.kwargs
        assert call_kwargs["extra_body"]["chat_template_kwargs"] == {
            "thinking_budget": 512,
            "enable_thinking": True,
        }


@pytest.mark.asyncio
async def test_reasoning_effort_for_openai_model(tmp_path):
    instructions = {
        "TestAssistant": {
            "model": "gpt-5.4",
            "temperature": 0.5,
            "system_instructions": "You are helpful.",
            "reasoning": {"effort": "high"},
        }
    }
    builder = ChainBuilder(config=_make_config(tmp_path), instructions=instructions)

    with patch("agent.chains.ChatOpenAIWithReasoning") as mock_llm_class:
        await builder._get_llm_for_assistant("TestAssistant", streaming=False)
        call_kwargs = mock_llm_class.call_args.kwargs
        assert call_kwargs["extra_body"] == {"reasoning": {"effort": "high"}}


@pytest.mark.asyncio
async def test_agent_llm_receives_translated_reasoning(tmp_path):
    """The agent path now translates reasoning (regression: it used to drop it)."""
    instructions = {
        "TestAssistant": {
            "model": "Qwen3-8B",
            "temperature": 0.5,
            "system_instructions": "You are helpful.",
            "reasoning": {"enabled": False},
        }
    }
    builder = ChainBuilder(config=_make_config(tmp_path), instructions=instructions)

    with patch("agent.chains.ChatOpenAIAgentLLM") as mock_agent_llm:
        await builder._get_agent_llm("TestAssistant")
        call_kwargs = mock_agent_llm.call_args.kwargs
        assert call_kwargs["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}
        }


@pytest.mark.asyncio
async def test_agent_llm_cache_key_includes_reasoning(tmp_path):
    instructions = {
        "TestAssistant": {
            "model": "Qwen3-8B",
            "temperature": 0.5,
            "system_instructions": "You are helpful.",
            "reasoning": {"enabled": False},
        },
        "Other": {
            "model": "Qwen3-8B",
            "temperature": 0.5,
            "system_instructions": "You are helpful.",
            "reasoning": {"enabled": True},
        },
    }
    builder = ChainBuilder(config=_make_config(tmp_path), instructions=instructions)

    with patch("agent.chains.ChatOpenAIAgentLLM") as mock_agent_llm:
        await builder._get_agent_llm("TestAssistant")
        await builder._get_agent_llm("Other")
        assert mock_agent_llm.call_count == 2
        kwargs_a = mock_agent_llm.call_args_list[0].kwargs
        kwargs_b = mock_agent_llm.call_args_list[1].kwargs
        assert kwargs_a["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
        assert kwargs_b["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


# ---------------------------------------------------------------------------
# _resolve_reasoning_rule — probed capabilities override family matching
# ---------------------------------------------------------------------------


class _FakeReasoningProviderStore:
    def __init__(self, caps_by_key=None, default_provider=None):
        self._caps = caps_by_key or {}
        self._default = default_provider

    async def get_model_reasoning_status(self, provider_id, model_id):
        return self._caps.get((provider_id, model_id))

    async def get_default_provider(self):
        return self._default


class _FakeProvider:
    def __init__(self, pid):
        self.id = pid
        self.base_url = "http://llm"
        self.timeout_seconds = 30


def _make_builder(tmp_path, provider_store=None):
    instructions = {
        "A": {"model": "m", "temperature": 0.5, "system_instructions": "x"}
    }
    builder = ChainBuilder(config=_make_config(tmp_path), instructions=instructions)
    if provider_store is not None:
        builder.set_provider_store(provider_store)
    return builder


@pytest.mark.asyncio
async def test_resolve_reasoning_rule_prefers_probed_caps(tmp_path):
    """A generic-named model with cached probed caps gets the probed wire."""
    store = _FakeReasoningProviderStore(
        caps_by_key={
            ("p1", "custom-model"): {
                "capabilities": {
                    "supports_effort": True,
                    "supports_thinking_toggle": False,
                },
                "check_error": None,
            }
        }
    )
    builder = _make_builder(tmp_path, provider_store=store)
    rule = await builder._resolve_reasoning_rule("custom-model", "http://llm/v1", "p1")
    assert rule["family"] == "probed"
    assert rule["wire"] == "probed_effort"


@pytest.mark.asyncio
async def test_resolve_reasoning_rule_falls_back_to_family(tmp_path):
    """No cached caps → regex family matching decides."""
    store = _FakeReasoningProviderStore()
    builder = _make_builder(tmp_path, provider_store=store)
    rule = await builder._resolve_reasoning_rule("Qwen3-8B", "http://llm/v1", "p1")
    assert rule["family"] == "qwen_thinking"


@pytest.mark.asyncio
async def test_resolve_reasoning_rule_no_store_uses_family(tmp_path):
    builder = _make_builder(tmp_path, provider_store=None)
    rule = await builder._resolve_reasoning_rule("custom-model", "http://llm/v1", None)
    assert rule["family"] == "generic"


@pytest.mark.asyncio
async def test_resolve_reasoning_rule_checks_default_provider(tmp_path):
    """Default-provider caps apply when no explicit provider is given."""
    store = _FakeReasoningProviderStore(
        caps_by_key={
            ("pdefault", "m"): {
                "capabilities": {
                    "supports_effort": False,
                    "supports_thinking_toggle": True,
                },
                "check_error": None,
            }
        },
        default_provider=_FakeProvider("pdefault"),
    )
    builder = _make_builder(tmp_path, provider_store=store)
    rule = await builder._resolve_reasoning_rule("m", "http://llm/v1", None)
    assert rule["wire"] == "qwen_thinking"


@pytest.mark.asyncio
async def test_resolve_reasoning_rule_probe_cannot_override_claude(tmp_path):
    """A probe result must NOT supersede richer families — claude's
    thinking.type/budget_tokens wire cannot be described by effort/toggle
    probe dimensions."""
    store = _FakeReasoningProviderStore(
        caps_by_key={
            ("p1", "claude-3-opus"): {
                "capabilities": {
                    "supports_effort": True,
                    "supports_thinking_toggle": True,
                },
                "check_error": None,
            }
        }
    )
    builder = _make_builder(tmp_path, provider_store=store)
    rule = await builder._resolve_reasoning_rule("claude-3-opus", "http://llm/v1", "p1")
    assert rule["wire"] == "claude_thinking"


@pytest.mark.asyncio
async def test_resolve_reasoning_rule_probe_corrects_family(tmp_path):
    """Probe disproves the family match: qwen3.8-name on an endpoint that
    rejects reasoning_effort but accepts enable_thinking → toggle wire."""
    store = _FakeReasoningProviderStore(
        caps_by_key={
            ("p1", "qwen3.8-flash"): {
                "capabilities": {
                    "supports_effort": False,
                    "supports_thinking_toggle": True,
                },
                "check_error": None,
            }
        }
    )
    builder = _make_builder(tmp_path, provider_store=store)
    rule = await builder._resolve_reasoning_rule("qwen3.8-flash", "http://llm/v1", "p1")
    assert rule["wire"] == "qwen_thinking"


@pytest.mark.asyncio
async def test_build_chains_system_instructions_with_braces_stay_literal(tmp_path):
    """JSON examples and placeholders in system_instructions must not be
    parsed as template variables — a ("system", str) tuple turns them into
    input_variables and crashes format_messages with KeyError."""
    raw_instructions = (
        'Du bist hilfsbereit.\n\n'
        '```json\n{"id": "mail_01", "category": "legitim", "task": {"type": "save"}}\n```\n\n'
        'Kontext: {context} und {{context}} bleiben literal.'
    )
    instructions = {
        "TestAssistant": {
            "model": "gpt-4",
            "temperature": 0.5,
            "system_instructions": raw_instructions,
        }
    }
    config = AppConfig(
        data_dir=tmp_path / "data",
        instructions_file=tmp_path / "instructions.yaml",
        pg_host="localhost",
        pg_port=5432,
        pg_database="test",
        pg_user="test",
        pg_password="test",
    )
    builder = ChainBuilder(config=config, instructions=instructions)

    real_prompts = []
    orig_from_messages = ChatPromptTemplate.from_messages

    def _spy(messages):
        # Build the real template for assertion capture, but hand the chain the
        # fake — piping a real ChatPromptTemplate into the object() LLM mock
        # raises TypeError in coerce_to_runnable.
        real_prompts.append(orig_from_messages(messages))
        return _FakePromptTemplate(messages)

    with patch.object(ChatPromptTemplate, "from_messages", side_effect=_spy):
        with patch.object(
            builder, "_get_llm_for_assistant", new_callable=AsyncMock, return_value=object()
        ):
            await builder.build_chains(
                "TestAssistant",
                use_contextualize=False,
                retriever=None,
                retrieval_degraded=True,
                rag_context="",
            )

    qa_prompt = next(
        t for t in real_prompts
        if t.messages and isinstance(t.messages[0], SystemMessage)
    )
    # No brace content leaked into template variables.
    assert '"id"' not in qa_prompt.input_variables
    assert "context" not in qa_prompt.input_variables
    # Formatting must not raise — and the text arrives verbatim.
    msgs = qa_prompt.format_messages(chat_history=[], input_messages=[])
    assert msgs[0].content == raw_instructions
