from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.history_budget import (
    SUMMARY_PREFIX,
    RollingHistorySettings,
    build_rolling_history,
    count_messages_tokens,
    resolve_rolling_history_settings,
)


class FakeSummaryLLM:
    async def ainvoke(self, payload):
        return AIMessage(content="Der Nutzer fragte zuvor nach Phishing, Backups und Incident-Kommunikation.")


def _settings(**overrides) -> RollingHistorySettings:
    values = {
        "enabled": True,
        "context_window": 500,
        "max_output_tokens": 80,
        "rag_reserved_tokens": 50,
        "min_recent_turns": 2,
        "safety_margin_ratio": 0.02,
        "summary_timeout_seconds": 2.0,
        "model": "gpt-4o-mini",
        "system_instructions": "Antworte kurz. Wissen: {{context}}",
    }
    values.update(overrides)
    return RollingHistorySettings(**values)


@pytest.mark.asyncio
async def test_build_rolling_history_keeps_history_when_budget_fits():
    history = [HumanMessage(content="Hallo"), AIMessage(content="Wie kann ich helfen?")]

    result = await build_rolling_history(
        history,
        HumanMessage(content="Was ist Phishing?"),
        settings=_settings(context_window=2000),
        summary_llm_factory=lambda: FakeSummaryLLM(),
    )

    assert result == history
    assert all(SUMMARY_PREFIX not in str(message.content) for message in result)


@pytest.mark.asyncio
async def test_build_rolling_history_summarizes_only_overflow_and_keeps_recent_turns():
    history = []
    for index in range(8):
        history.append(HumanMessage(content=f"Alte Nutzerfrage {index}: " + "Phishing Backup " * 20))
        history.append(AIMessage(content=f"Alte Antwort {index}: " + "Incident Response " * 20))

    result = await build_rolling_history(
        history,
        HumanMessage(content="Was soll ich jetzt zuerst tun?"),
        settings=_settings(context_window=900),
        summary_llm_factory=lambda: FakeSummaryLLM(),
    )

    assert any(isinstance(message, HumanMessage) and str(message.content).startswith(SUMMARY_PREFIX) for message in result)
    assert result[-4:] == history[-4:]
    assert count_messages_tokens(result, "gpt-4o-mini") <= 900


@pytest.mark.asyncio
async def test_build_rolling_history_summarizes_when_factory_returns_task():
    history = []
    for index in range(8):
        history.append(HumanMessage(content=f"Alte Nutzerfrage {index}: " + "Phishing Backup " * 20))
        history.append(AIMessage(content=f"Alte Antwort {index}: " + "Incident Response " * 20))

    async def _get_llm():
        return FakeSummaryLLM()

    result = await build_rolling_history(
        history,
        HumanMessage(content="Was soll ich jetzt zuerst tun?"),
        settings=_settings(context_window=900),
        summary_llm_factory=lambda: asyncio.create_task(_get_llm()),
    )

    assert any(isinstance(message, HumanMessage) and str(message.content).startswith(SUMMARY_PREFIX) for message in result)
    assert result[-4:] == history[-4:]
    assert count_messages_tokens(result, "gpt-4o-mini") <= 900


@pytest.mark.asyncio
async def test_build_rolling_history_falls_back_to_trimming_when_summary_fails():
    class FailingLLM:
        async def ainvoke(self, payload):
            raise RuntimeError("boom")

    history = [HumanMessage(content="alt " * 100), AIMessage(content="antwort " * 100)] * 4

    result = await build_rolling_history(
        history,
        HumanMessage(content="weiter"),
        settings=_settings(),
        summary_llm_factory=lambda: FailingLLM(),
    )

    assert all(SUMMARY_PREFIX not in str(message.content) for message in result)
    assert count_messages_tokens(result, "gpt-4o-mini") <= 500


def test_resolve_rolling_history_settings_uses_assistant_overrides(app_config):
    app_config.rolling_history_context_window = 16000
    assistant = {
        "model": "gpt-oss:120b",
        "system_instructions": "Wissen: {{context}}",
        "context_window": 64000,
        "max_output_tokens": 4096,
        "rag_reserved_tokens": 2048,
        "min_recent_turns": 3,
    }

    settings = resolve_rolling_history_settings(app_config, assistant)

    assert settings.context_window == 64000
    assert settings.max_output_tokens == 4096
    assert settings.rag_reserved_tokens == 2048
    assert settings.min_recent_turns == 3
    assert settings.model == "gpt-oss:120b"
