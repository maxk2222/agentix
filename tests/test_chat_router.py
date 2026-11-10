from __future__ import annotations

from types import SimpleNamespace

import asyncio
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.routers.chat import chat_endpoint, chat_run_cancel
from app.chat_runs import ActiveChatRun, _runs, get_run, register_run
from app.schemas import ChatRequest, Message


class FakeMessageStore:
    def __init__(self) -> None:
        self.saved: list[dict] = []

    async def save_turn(self, **kwargs) -> None:
        self.saved.append(kwargs)


class FakeRAGApplication:
    ready = True
    instructions = {
        "CyberSecurityAssistant": {
            "model": "gpt-4o-mini",
            # Explicit per-assistant override wins over provider metadata and
            # the model-family map — keeps the rolling-history budget testable.
            "context_window": 420,
            "system_instructions": "Antworte kurz. Wissen: {{context}}",
        }
    }
    engine = object()

    def __init__(self, usage_metadata=None, reasoning_content=None) -> None:
        self.last_chat_history = None
        self._usage_metadata = usage_metadata
        self._reasoning_content = reasoning_content

    async def refresh_instructions_if_needed(self) -> bool:
        return False

    async def run(self, user_message, *, chat_history, model_name, retrieval_info=None, max_tokens=None, **kwargs):
        self.last_chat_history = chat_history
        if retrieval_info is not None:
            retrieval_info.update(
                {
                    "retrieval_query": user_message,
                    "rag_context": [{"content": "Kontext", "metadata": {"source": "test.md"}}],
                    "status": "ok",
                }
            )
            if self._usage_metadata is not None:
                retrieval_info["_llm_usage_metadata"] = self._usage_metadata
        result = {"answer": f"Echo: {user_message}"}
        if self._reasoning_content is not None:
            result["additional_kwargs"] = {"reasoning_content": self._reasoning_content}
        return result

    async def get_metadata_llm(self, model_name=None, purpose=None):
        return None

    async def run_streaming(self, user_message, *, chat_history, model_name, retrieval_info=None, max_tokens=None, **kwargs):
        self.last_chat_history = chat_history
        if retrieval_info is not None:
            retrieval_info.update(
                {
                    "status": "ok",
                    "chunks_retrieved": 1,
                    "rag_context_tokens": 0,
                    "rag_context": [],
                }
            )
            if self._usage_metadata is not None:
                retrieval_info["_llm_usage_metadata"] = self._usage_metadata
        # Small real delay so measured response_time_ms is realistic — an
        # instant stream produces sub-ms timings where no sane tps exists.
        await asyncio.sleep(0.02)
        yield {"content": f"Echo: {user_message}"}


@pytest.mark.asyncio
async def test_non_streaming_chat_persists_user_and_assistant_in_one_turn(monkeypatch):
    store = FakeMessageStore()
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                rag_application=FakeRAGApplication(),
                app_config=None,
                _message_store=store,
            )
        ),
        state=SimpleNamespace(),
        headers={"content-type": "application/json"},
    )
    chat_request = ChatRequest(
        model="CyberSecurityAssistant",
        messages=[Message(role="user", content="Hallo")],
        stream=False,
    )

    monkeypatch.setattr(
        "app.routers.chat.PublicChatConfigStore",
        lambda engine: FakePublicChatConfigStore(
            {"enabled": True, "enabled_assistants": ["CyberSecurityAssistant"]}
        ),
    )
    monkeypatch.setattr(
        "app.chat_group_access.ChatRoleAssistantsStore",
        lambda engine: FakeChatRoleAssistantsStore(["CyberSecurityAssistant"]),
    )

    response = await chat_endpoint(chat_request, request)

    assert response.status_code == 200
    assert len(store.saved) == 1
    assert store.saved[0]["user_content"] == "Hallo"
    assert store.saved[0]["assistant_content"] == "Echo: Hallo"
    assert store.saved[0]["chat_history"] == []
    assert store.saved[0]["retrieval_info"]["status"] == "ok"


@pytest.mark.asyncio
async def test_non_streaming_chat_accepts_client_provided_conversation_id(monkeypatch):
    store = FakeMessageStore()
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                rag_application=FakeRAGApplication(),
                app_config=SimpleNamespace(dashboard_session_secret="test", stream_idle_timeout_seconds=30.0),
                _message_store=store,
            )
        ),
        state=SimpleNamespace(),
        headers={"content-type": "application/json"},
        cookies={},
    )
    chat_request = ChatRequest(
        model="CyberSecurityAssistant",
        messages=[Message(role="user", content="Bonjour")],
        stream=False,
        conversation_id="client-12345",
    )

    monkeypatch.setattr(
        "app.routers.chat.PublicChatConfigStore",
        lambda engine: FakePublicChatConfigStore(
            {"enabled": True, "enabled_assistants": ["CyberSecurityAssistant"]}
        ),
    )
    monkeypatch.setattr(
        "app.chat_group_access.ChatRoleAssistantsStore",
        lambda engine: FakeChatRoleAssistantsStore(["CyberSecurityAssistant"]),
    )

    response = await chat_endpoint(chat_request, request)

    assert response.status_code == 200
    assert store.saved[0]["conversation_id"] == "client-12345"


@pytest.mark.asyncio
async def test_non_streaming_chat_applies_rolling_history_budget(monkeypatch):
    rag_app = FakeRAGApplication()
    config = SimpleNamespace(
        max_history_messages=0,
        rolling_history_enabled=True,
        rolling_history_context_window=420,
        rolling_history_max_output_tokens=80,
        rolling_history_rag_reserved_tokens=40,
        rolling_history_min_recent_turns=1,
        rolling_history_safety_margin_ratio=0.02,
        rolling_history_summary_timeout_seconds=0.1,
        vector_search_k=0,
        chunk_size=750,
        max_messages=100,
        max_message_chars=20_000,
        max_total_chars=120_000,
        require_last_user=True,
        allow_non_string_content=False,
        validation_control_chars_mode="reject",
        validation_enforce=True,
        stream_idle_timeout_seconds=5.0,
        dashboard_session_secret="test",
    )
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                rag_application=rag_app,
                app_config=config,
                _message_store=FakeMessageStore(),
            )
        ),
        state=SimpleNamespace(),
        headers={"content-type": "application/json"},
        cookies={},
    )
    messages = []
    for index in range(8):
        messages.append(Message(role="user", content=f"Frage {index} " + "Phishing " * 30))
        messages.append(Message(role="assistant", content=f"Antwort {index} " + "Backup " * 30))
    messages.append(Message(role="user", content="Was jetzt?"))
    chat_request = ChatRequest(
        model="CyberSecurityAssistant",
        messages=messages,
        stream=False,
    )

    monkeypatch.setattr(
        "app.routers.chat.PublicChatConfigStore",
        lambda engine: FakePublicChatConfigStore(
            {"enabled": True, "enabled_assistants": ["CyberSecurityAssistant"]}
        ),
    )
    monkeypatch.setattr(
        "app.chat_group_access.ChatRoleAssistantsStore",
        lambda engine: FakeChatRoleAssistantsStore(["CyberSecurityAssistant"]),
    )

    response = await chat_endpoint(chat_request, request)

    assert response.status_code == 200
    assert rag_app.last_chat_history is not None
    assert len(rag_app.last_chat_history) < len(messages) - 1


@pytest.mark.asyncio
async def test_non_streaming_chat_persists_rolled_history_summary(monkeypatch):
    store = FakeMessageStore()
    config = SimpleNamespace(
        max_history_messages=10,
        rolling_history_enabled=True,
        rolling_history_context_window=4096,
        rolling_history_max_output_tokens=256,
        rolling_history_rag_reserved_tokens=128,
        rolling_history_min_recent_turns=1,
        rolling_history_safety_margin_ratio=0.02,
        rolling_history_summary_timeout_seconds=0.1,
        vector_search_k=0,
        chunk_size=750,
        stream_idle_timeout_seconds=5.0,
        dashboard_session_secret="test",
    )
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                rag_application=FakeRAGApplication(),
                app_config=config,
                _message_store=store,
            )
        ),
        state=SimpleNamespace(),
        headers={"content-type": "application/json"},
        cookies={},
    )
    chat_request = ChatRequest(
        model="CyberSecurityAssistant",
        messages=[Message(role="user", content="Frage 1"), Message(role="assistant", content="Antwort 1"), Message(role="user", content="Letzte Frage")],
        stream=False,
    )

    async def fake_build_rolling_history(history, latest_user_message, *, settings, summary_llm_factory=None):
        return [
            SystemMessage(content="System prompt"),
            HumanMessage(content="BISHERIGE ZUSAMMENFASSUNG DES ÄLTEREN CHATVERLAUFS:\nDiese Nachricht wurde automatisch aus älteren Chatnachrichten komprimiert und dient nur als Kontext.\n\n<summary>\nAlte Frage und Antwort zusammengefasst\n</summary>"),
            AIMessage(content="Antwort auf die letzte Frage"),
        ]

    monkeypatch.setattr("app.routers.chat.build_rolling_history", fake_build_rolling_history)

    monkeypatch.setattr(
        "app.routers.chat.PublicChatConfigStore",
        lambda engine: FakePublicChatConfigStore(
            {"enabled": True, "enabled_assistants": ["CyberSecurityAssistant"]}
        ),
    )
    monkeypatch.setattr(
        "app.chat_group_access.ChatRoleAssistantsStore",
        lambda engine: FakeChatRoleAssistantsStore(["CyberSecurityAssistant"]),
    )

    response = await chat_endpoint(chat_request, request)

    assert response.status_code == 200
    assert any(
        entry["content"].startswith("BISHERIGE ZUSAMMENFASSUNG DES ÄLTEREN CHATVERLAUFS:")
        for entry in store.saved[0]["chat_history"]
    )
    assert not any(entry["content"] == "Letzte Frage" for entry in store.saved[0]["chat_history"])


@pytest.mark.asyncio
async def test_non_streaming_chat_uses_provider_usage_metadata(monkeypatch):
    store = FakeMessageStore()
    usage_meta = {
        "prompt_tokens": 15,
        "completion_tokens": 25,
        "total_tokens": 40,
        "reasoning_tokens": 8,
    }
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                rag_application=FakeRAGApplication(usage_metadata=usage_meta),
                app_config=SimpleNamespace(dashboard_session_secret="test", stream_idle_timeout_seconds=30.0),
                _message_store=store,
            )
        ),
        state=SimpleNamespace(),
        headers={"content-type": "application/json"},
        cookies={},
    )
    chat_request = ChatRequest(
        model="CyberSecurityAssistant",
        messages=[Message(role="user", content="Hallo")],
        stream=False,
    )

    monkeypatch.setattr(
        "app.routers.chat.PublicChatConfigStore",
        lambda engine: FakePublicChatConfigStore(
            {"enabled": True, "enabled_assistants": ["CyberSecurityAssistant"]}
        ),
    )
    monkeypatch.setattr(
        "app.chat_group_access.ChatRoleAssistantsStore",
        lambda engine: FakeChatRoleAssistantsStore(["CyberSecurityAssistant"]),
    )

    response = await chat_endpoint(chat_request, request)

    assert response.status_code == 200
    import json
    payload = json.loads(response.body)
    assert payload["usage"]["prompt_tokens"] == 15
    assert payload["usage"]["completion_tokens"] == 25
    assert payload["usage"]["total_tokens"] == 40
    assert payload["usage"]["reasoning_tokens"] == 8


@pytest.mark.asyncio
async def test_non_streaming_chat_fallback_reasoning_tokens_from_text(monkeypatch):
    store = FakeMessageStore()
    reasoning_text = "Let me think about this step by step. First, I need to analyze the greeting."
    usage_meta = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "reasoning_tokens": 0,
    }
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                rag_application=FakeRAGApplication(
                    usage_metadata=usage_meta,
                    reasoning_content=reasoning_text,
                ),
                app_config=SimpleNamespace(dashboard_session_secret="test", stream_idle_timeout_seconds=30.0),
                _message_store=store,
            )
        ),
        state=SimpleNamespace(),
        headers={"content-type": "application/json"},
        cookies={},
    )
    chat_request = ChatRequest(
        model="CyberSecurityAssistant",
        messages=[Message(role="user", content="Hallo")],
        stream=False,
    )

    monkeypatch.setattr(
        "app.routers.chat.PublicChatConfigStore",
        lambda engine: FakePublicChatConfigStore(
            {"enabled": True, "enabled_assistants": ["CyberSecurityAssistant"]}
        ),
    )
    monkeypatch.setattr(
        "app.chat_group_access.ChatRoleAssistantsStore",
        lambda engine: FakeChatRoleAssistantsStore(["CyberSecurityAssistant"]),
    )

    response = await chat_endpoint(chat_request, request)

    assert response.status_code == 200
    import json
    payload = json.loads(response.body)
    # reasoning_tokens should be > 0 because fallback counted tokens from reasoning_text
    assert payload["usage"]["reasoning_tokens"] > 0
    assert payload["usage"]["reasoning_tokens"] != 0


class FakeRAGApplicationInternalOnly(FakeRAGApplication):
    instructions = {
        "InternalAssistant": {
            "model": "gpt-4o-mini",
            "system_instructions": "Internal only.",
            "api_accessible": False,
        }
    }


class FakePublicChatConfigStore:
    def __init__(self, config: dict | None = None) -> None:
        self._config = config

    async def get(self):
        return self._config

    async def save(self, config: dict, created_by: str = "admin") -> None:
        pass


class FakeChatRoleAssistantsStore:
    def __init__(self, assistants: list[str] | None = None) -> None:
        self._assistants = assistants or []

    async def get_assistants_for_role(self, role: str):
        return list(self._assistants)


def _internal_request(store, *, headers=None, cookies=None):
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                rag_application=FakeRAGApplicationInternalOnly(),
                app_config=SimpleNamespace(
                    dashboard_session_secret="test",
                    stream_idle_timeout_seconds=30.0,
                    max_messages=100,
                    max_message_chars=20_000,
                    max_total_chars=120_000,
                    require_last_user=True,
                    allow_non_string_content=False,
                    validation_control_chars_mode="reject",
                    validation_enforce=True,
                ),
                _message_store=store,
            )
        ),
        state=SimpleNamespace(),
        headers=headers or {"content-type": "application/json"},
        cookies=cookies or {},
        query_params={},
    )


def _pc_token(role: str = "standard") -> str:
    from app.routers.public_chat_auth import _create_public_chat_access_token

    return _create_public_chat_access_token(
        SimpleNamespace(dashboard_session_secret="test"), role
    )


@pytest.mark.asyncio
async def test_chat_endpoint_denies_internal_assistant_anonymously(monkeypatch):
    """Anonymous callers cannot reach api_accessible=false assistants, even
    when the public chat feature is enabled for them."""
    store = FakeMessageStore()
    request = _internal_request(store)
    chat_request = ChatRequest(
        model="InternalAssistant",
        messages=[Message(role="user", content="Hallo")],
        stream=False,
    )

    monkeypatch.setattr(
        "app.routers.chat.PublicChatConfigStore",
        lambda engine: FakePublicChatConfigStore(
            {"enabled": True, "enabled_assistants": ["InternalAssistant"]}
        ),
    )
    monkeypatch.setattr(
        "app.chat_group_access.ChatRoleAssistantsStore",
        lambda engine: FakeChatRoleAssistantsStore(["InternalAssistant"]),
    )

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await chat_endpoint(chat_request, request)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_chat_endpoint_allows_internal_assistant_with_public_chat_token(monkeypatch):
    """A valid scoped public-chat token opens api_accessible=false assistants
    that are allowed for the token's role."""
    store = FakeMessageStore()
    headers = {"content-type": "application/json", "x-public-chat-token": _pc_token()}
    request = _internal_request(store, headers=headers)
    chat_request = ChatRequest(
        model="InternalAssistant",
        messages=[Message(role="user", content="Hallo")],
        stream=False,
    )

    monkeypatch.setattr(
        "app.routers.chat.PublicChatConfigStore",
        lambda engine: FakePublicChatConfigStore(
            {"enabled": True, "enabled_assistants": ["InternalAssistant"]}
        ),
    )
    monkeypatch.setattr(
        "app.chat_group_access.ChatRoleAssistantsStore",
        lambda engine: FakeChatRoleAssistantsStore(["InternalAssistant"]),
    )

    response = await chat_endpoint(chat_request, request)

    assert response.status_code == 200
    assert len(store.saved) == 1
    assert store.saved[0]["user_content"] == "Hallo"


@pytest.mark.asyncio
async def test_chat_endpoint_rejects_tampered_public_chat_token(monkeypatch):
    """A forged token signature is ignored → caller is treated as anonymous."""
    store = FakeMessageStore()
    forged = _pc_token()[:-4] + "AAAA"
    headers = {"content-type": "application/json", "x-public-chat-token": forged}
    request = _internal_request(store, headers=headers)
    chat_request = ChatRequest(
        model="InternalAssistant",
        messages=[Message(role="user", content="Hallo")],
        stream=False,
    )

    monkeypatch.setattr(
        "app.routers.chat.PublicChatConfigStore",
        lambda engine: FakePublicChatConfigStore(
            {"enabled": True, "enabled_assistants": ["InternalAssistant"]}
        ),
    )
    monkeypatch.setattr(
        "app.chat_group_access.ChatRoleAssistantsStore",
        lambda engine: FakeChatRoleAssistantsStore(["InternalAssistant"]),
    )

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await chat_endpoint(chat_request, request)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_chat_endpoint_token_denied_when_public_chat_disabled(monkeypatch):
    """enabled=false turns the whole feature off — token cannot bypass."""
    store = FakeMessageStore()
    headers = {"content-type": "application/json", "x-public-chat-token": _pc_token()}
    request = _internal_request(store, headers=headers)
    chat_request = ChatRequest(
        model="InternalAssistant",
        messages=[Message(role="user", content="Hallo")],
        stream=False,
    )

    monkeypatch.setattr(
        "app.routers.chat.PublicChatConfigStore",
        lambda engine: FakePublicChatConfigStore({"enabled": False}),
    )
    monkeypatch.setattr(
        "app.chat_group_access.ChatRoleAssistantsStore",
        lambda engine: FakeChatRoleAssistantsStore(["InternalAssistant"]),
    )

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await chat_endpoint(chat_request, request)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_chat_endpoint_token_respects_role_allowlist(monkeypatch):
    """Token role 'standard' is not allowed the internal assistant when its
    role mapping lists a different model."""
    store = FakeMessageStore()
    headers = {"content-type": "application/json", "x-public-chat-token": _pc_token()}
    request = _internal_request(store, headers=headers)
    chat_request = ChatRequest(
        model="InternalAssistant",
        messages=[Message(role="user", content="Hallo")],
        stream=False,
    )

    monkeypatch.setattr(
        "app.routers.chat.PublicChatConfigStore",
        lambda engine: FakePublicChatConfigStore({"enabled": True}),
    )
    monkeypatch.setattr(
        "app.chat_group_access.ChatRoleAssistantsStore",
        lambda engine: FakeChatRoleAssistantsStore(["OtherAssistant"]),
    )

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await chat_endpoint(chat_request, request)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_chat_endpoint_allows_internal_assistant_for_logged_in_chat_user(monkeypatch):
    store = FakeMessageStore()
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                rag_application=FakeRAGApplicationInternalOnly(),
                app_config=SimpleNamespace(
                    dashboard_session_secret="test",
                    stream_idle_timeout_seconds=30.0,
                    max_messages=100,
                    max_message_chars=20_000,
                    max_total_chars=120_000,
                    require_last_user=True,
                    allow_non_string_content=False,
                    validation_control_chars_mode="reject",
                    validation_enforce=True,
                ),
                _message_store=store,
            )
        ),
        state=SimpleNamespace(),
        headers={"content-type": "application/json"},
        cookies={},
    )
    chat_request = ChatRequest(
        model="InternalAssistant",
        messages=[Message(role="user", content="Hallo")],
        stream=False,
    )

    async def fake_get_current_chat_user(request):
        return {
            "id": "chat-user-123",
            "username": "testuser",
            "display_name": "Test User",
            "role": "standard",
            "session_version": 1,
        }

    monkeypatch.setattr(
        "app.routers.chat._get_current_chat_user", fake_get_current_chat_user
    )
    monkeypatch.setattr(
        "app.routers.chat.PublicChatConfigStore",
        lambda engine: FakePublicChatConfigStore(
            {"enabled": True, "enabled_assistants": ["InternalAssistant"]}
        ),
    )
    monkeypatch.setattr(
        "app.chat_group_access.ChatRoleAssistantsStore",
        lambda engine: FakeChatRoleAssistantsStore(["InternalAssistant"]),
    )

    response = await chat_endpoint(chat_request, request)

    assert response.status_code == 200
    assert len(store.saved) == 1
    assert store.saved[0]["user_content"] == "Hallo"


@pytest.mark.asyncio
async def test_chat_endpoint_allows_internal_assistant_with_dashboard_session(monkeypatch):
    store = FakeMessageStore()
    config = SimpleNamespace(
        dashboard_session_secret="test-secret",
        dashboard_session_ttl_seconds=3600,
        max_history_messages=0,
        rolling_history_enabled=False,
        rolling_history_context_window=4096,
        rolling_history_max_output_tokens=256,
        rolling_history_rag_reserved_tokens=128,
        rolling_history_min_recent_turns=1,
        rolling_history_safety_margin_ratio=0.02,
        rolling_history_summary_timeout_seconds=0.1,
        vector_search_k=0,
        chunk_size=750,
        stream_idle_timeout_seconds=5.0,
        max_messages=100,
        max_message_chars=20_000,
        max_total_chars=120_000,
        require_last_user=True,
        allow_non_string_content=False,
        validation_control_chars_mode="reject",
        validation_enforce=True,
    )
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                rag_application=FakeRAGApplicationInternalOnly(),
                app_config=config,
                _message_store=store,
            )
        ),
        state=SimpleNamespace(),
        headers={"content-type": "application/json"},
        cookies={"dashboard_session": "fake-cookie"},
    )
    chat_request = ChatRequest(
        model="InternalAssistant",
        messages=[Message(role="user", content="Hallo")],
        stream=False,
    )

    def _fake_read_session_cookie(req, cfg):
        return {"role": "admin"}

    monkeypatch.setattr("app.routers.chat.read_session_cookie", _fake_read_session_cookie)

    response = await chat_endpoint(chat_request, request)

    assert response.status_code == 200
    assert len(store.saved) == 1
    assert store.saved[0]["user_content"] == "Hallo"


class FakeChainEntry:
    def __init__(self, chain_id: str, name: str) -> None:
        self.id = chain_id
        self.name = name
        self.description = ""
        self.enabled = True
        self.api_accessible = True
        self.flow_definition = {"nodes": [], "edges": []}
        self.execution_options = {}
        self.created_at = None
        self.updated_at = None


class FakeAgentChainsStore:
    def __init__(self, entries: list[FakeChainEntry]) -> None:
        self._entries = entries
        self._tables_ensured = False

    async def ensure_tables(self) -> None:
        self._tables_ensured = True

    async def get_chain(self, chain_id: str):
        for entry in self._entries:
            if entry.id == chain_id:
                return entry
        return None

    async def get_chain_by_name(self, name: str):
        for entry in self._entries:
            if entry.name == name:
                return entry
        return None


class FakeRAGApplicationWithChain(FakeRAGApplication):
    async def run(self, user_message, *, chat_history, model_name, retrieval_info=None, max_tokens=None, **kwargs):
        if retrieval_info is not None:
            retrieval_info.update({"retrieval_query": user_message, "rag_context": [], "status": "ok"})
        return {"answer": f"ChainEcho: {user_message}"}

    async def run_chain(self, *, chain_definition, input_text, chat_history=None, conversation_id=None, **kwargs):
        return {"answer": f"ChainEcho: {input_text}"}


@pytest.mark.asyncio
async def test_chat_endpoint_resolves_chain_by_name(monkeypatch):
    store = FakeMessageStore()
    chain_entry = FakeChainEntry(chain_id="chain-uuid-123", name="TestChain")
    fake_store = FakeAgentChainsStore([chain_entry])

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                rag_application=FakeRAGApplicationWithChain(),
                app_config=SimpleNamespace(dashboard_session_secret="test", stream_idle_timeout_seconds=30.0),
                _message_store=store,
            )
        ),
        state=SimpleNamespace(),
        headers={"content-type": "application/json"},
        cookies={},
    )
    chat_request = ChatRequest(
        model="chain:TestChain",
        messages=[Message(role="user", content="Hallo")],
        stream=False,
    )

    monkeypatch.setattr("app.routers.chat.AgentChainsStore", lambda engine: fake_store)

    monkeypatch.setattr(
        "app.routers.chat.PublicChatConfigStore",
        lambda engine: FakePublicChatConfigStore(
            {"enabled": True, "enabled_assistants": ["chain:TestChain"]}
        ),
    )
    monkeypatch.setattr(
        "app.chat_group_access.ChatRoleAssistantsStore",
        lambda engine: FakeChatRoleAssistantsStore(["chain:TestChain"]),
    )

    response = await chat_endpoint(chat_request, request)

    assert response.status_code == 200
    assert len(store.saved) == 1
    assert store.saved[0]["assistant_content"] == "ChainEcho: Hallo"


@pytest.mark.asyncio
async def test_chat_endpoint_resolves_chain_by_uuid(monkeypatch):
    store = FakeMessageStore()
    chain_entry = FakeChainEntry(chain_id="chain-uuid-456", name="UuidChain")
    fake_store = FakeAgentChainsStore([chain_entry])

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                rag_application=FakeRAGApplicationWithChain(),
                app_config=SimpleNamespace(dashboard_session_secret="test", stream_idle_timeout_seconds=30.0),
                _message_store=store,
            )
        ),
        state=SimpleNamespace(),
        headers={"content-type": "application/json"},
        cookies={},
    )
    chat_request = ChatRequest(
        model="chain:chain-uuid-456",
        messages=[Message(role="user", content="Hallo")],
        stream=False,
    )

    monkeypatch.setattr("app.routers.chat.AgentChainsStore", lambda engine: fake_store)

    monkeypatch.setattr(
        "app.routers.chat.PublicChatConfigStore",
        lambda engine: FakePublicChatConfigStore(
            {"enabled": True, "enabled_assistants": ["chain:chain-uuid-456"]}
        ),
    )
    monkeypatch.setattr(
        "app.chat_group_access.ChatRoleAssistantsStore",
        lambda engine: FakeChatRoleAssistantsStore(["chain:chain-uuid-456"]),
    )

    response = await chat_endpoint(chat_request, request)

    assert response.status_code == 200
    assert len(store.saved) == 1
    assert store.saved[0]["assistant_content"] == "ChainEcho: Hallo"


@pytest.mark.asyncio
async def test_chat_endpoint_returns_404_for_unknown_chain(monkeypatch):
    fake_store = FakeAgentChainsStore([])

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                rag_application=FakeRAGApplicationWithChain(),
                app_config=None,
                _message_store=FakeMessageStore(),
            )
        ),
        state=SimpleNamespace(),
        headers={"content-type": "application/json"},
    )
    chat_request = ChatRequest(
        model="chain:NonExistent",
        messages=[Message(role="user", content="Hallo")],
        stream=False,
    )

    monkeypatch.setattr("app.routers.chat.AgentChainsStore", lambda engine: fake_store)

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await chat_endpoint(chat_request, request)

    assert exc_info.value.status_code == 404
    assert "NonExistent" in exc_info.value.detail


@pytest.mark.asyncio
async def test_public_chat_filter_rejects_when_disabled(monkeypatch):
    """Internal assistant (api_accessible=false) without public chat → 403."""
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                rag_application=FakeRAGApplicationInternalOnly(),
                app_config=None,
                _message_store=FakeMessageStore(),
            )
        ),
        state=SimpleNamespace(),
        headers={"content-type": "application/json"},
    )
    chat_request = ChatRequest(
        model="InternalAssistant",
        messages=[Message(role="user", content="Hallo")],
        stream=False,
    )

    monkeypatch.setattr(
        "app.routers.chat.PublicChatConfigStore",
        lambda engine: FakePublicChatConfigStore({"enabled": False}),
    )
    monkeypatch.setattr(
        "app.chat_group_access.ChatRoleAssistantsStore",
        lambda engine: FakeChatRoleAssistantsStore([]),
    )

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await chat_endpoint(chat_request, request)

    assert exc_info.value.status_code == 403
    assert "not accessible" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_public_chat_filter_rejects_when_assistant_not_enabled(monkeypatch):
    """Internal assistant (api_accessible=false) with public chat ON but not in list → 403."""
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                rag_application=FakeRAGApplicationInternalOnly(),
                app_config=None,
                _message_store=FakeMessageStore(),
            )
        ),
        state=SimpleNamespace(),
        headers={"content-type": "application/json"},
    )
    chat_request = ChatRequest(
        model="InternalAssistant",
        messages=[Message(role="user", content="Hallo")],
        stream=False,
    )

    monkeypatch.setattr(
        "app.routers.chat.PublicChatConfigStore",
        lambda engine: FakePublicChatConfigStore(
            {"enabled": True, "enabled_assistants": ["OtherAssistant"]}
        ),
    )
    monkeypatch.setattr(
        "app.chat_group_access.ChatRoleAssistantsStore",
        lambda engine: FakeChatRoleAssistantsStore(["OtherAssistant"]),
    )

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await chat_endpoint(chat_request, request)

    assert exc_info.value.status_code == 403
    assert "accessible" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_public_chat_filter_allows_enabled_assistant(monkeypatch):
    store = FakeMessageStore()
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                rag_application=FakeRAGApplication(),
                app_config=None,
                _message_store=store,
            )
        ),
        state=SimpleNamespace(),
        headers={"content-type": "application/json"},
    )
    chat_request = ChatRequest(
        model="CyberSecurityAssistant",
        messages=[Message(role="user", content="Hallo")],
        stream=False,
    )

    monkeypatch.setattr(
        "app.routers.chat.PublicChatConfigStore",
        lambda engine: FakePublicChatConfigStore(
            {"enabled": True, "enabled_assistants": ["CyberSecurityAssistant"]}
        ),
    )
    monkeypatch.setattr(
        "app.chat_group_access.ChatRoleAssistantsStore",
        lambda engine: FakeChatRoleAssistantsStore(["CyberSecurityAssistant"]),
    )

    response = await chat_endpoint(chat_request, request)
    assert response.status_code == 200
    assert len(store.saved) == 1
    assert store.saved[0]["user_content"] == "Hallo"


@pytest.mark.asyncio
async def test_api_accessible_assistant_allowed_without_public_chat(monkeypatch):
    """api_accessible=true assistant should be reachable without public chat enabled."""
    store = FakeMessageStore()
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                rag_application=FakeRAGApplication(),
                app_config=None,
                _message_store=store,
            )
        ),
        state=SimpleNamespace(),
        headers={"content-type": "application/json"},
    )
    chat_request = ChatRequest(
        model="CyberSecurityAssistant",
        messages=[Message(role="user", content="Hallo")],
        stream=False,
    )

    monkeypatch.setattr(
        "app.routers.chat.PublicChatConfigStore",
        lambda engine: FakePublicChatConfigStore({"enabled": False}),
    )
    monkeypatch.setattr(
        "app.chat_group_access.ChatRoleAssistantsStore",
        lambda engine: FakeChatRoleAssistantsStore([]),
    )

    response = await chat_endpoint(chat_request, request)
    assert response.status_code == 200
    assert len(store.saved) == 1
    assert store.saved[0]["user_content"] == "Hallo"


@pytest.mark.asyncio
async def test_api_accessible_assistant_allowed_anonymously_regardless_of_allowlist(monkeypatch):
    """api_accessible is the single gate for anonymous callers — public-chat
    role allowlists no longer apply to them."""
    store = FakeMessageStore()
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                rag_application=FakeRAGApplication(),
                app_config=None,
                _message_store=store,
            )
        ),
        state=SimpleNamespace(),
        headers={"content-type": "application/json"},
    )
    chat_request = ChatRequest(
        model="CyberSecurityAssistant",
        messages=[Message(role="user", content="Hallo")],
        stream=False,
    )

    monkeypatch.setattr(
        "app.routers.chat.PublicChatConfigStore",
        lambda engine: FakePublicChatConfigStore(
            {"enabled": True, "enabled_assistants": ["OtherAssistant"]}
        ),
    )
    monkeypatch.setattr(
        "app.chat_group_access.ChatRoleAssistantsStore",
        lambda engine: FakeChatRoleAssistantsStore(["OtherAssistant"]),
    )

    response = await chat_endpoint(chat_request, request)
    assert response.status_code == 200
    assert len(store.saved) == 1


@pytest.mark.asyncio
async def test_api_accessible_assistant_allowed_when_in_anonymous_allowlist(monkeypatch):
    """api_accessible=true assistant in the anonymous allowlist is reachable."""
    store = FakeMessageStore()
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                rag_application=FakeRAGApplication(),
                app_config=None,
                _message_store=store,
            )
        ),
        state=SimpleNamespace(),
        headers={"content-type": "application/json"},
    )
    chat_request = ChatRequest(
        model="CyberSecurityAssistant",
        messages=[Message(role="user", content="Hallo")],
        stream=False,
    )

    monkeypatch.setattr(
        "app.routers.chat.PublicChatConfigStore",
        lambda engine: FakePublicChatConfigStore(
            {"enabled": True, "enabled_assistants": ["CyberSecurityAssistant"]}
        ),
    )
    monkeypatch.setattr(
        "app.chat_group_access.ChatRoleAssistantsStore",
        lambda engine: FakeChatRoleAssistantsStore(["CyberSecurityAssistant"]),
    )

    response = await chat_endpoint(chat_request, request)
    assert response.status_code == 200
    assert len(store.saved) == 1
    assert store.saved[0]["user_content"] == "Hallo"


@pytest.mark.asyncio
async def test_public_chat_filter_bypassed_with_api_key(monkeypatch):
    store = FakeMessageStore()
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                rag_application=FakeRAGApplication(),
                app_config=None,
                _message_store=store,
            )
        ),
        state=SimpleNamespace(api_key_id="test-key"),
        headers={"content-type": "application/json"},
    )
    chat_request = ChatRequest(
        model="CyberSecurityAssistant",
        messages=[Message(role="user", content="Hallo")],
        stream=False,
    )

    # Even with public chat disabled, API key should bypass
    monkeypatch.setattr(
        "app.routers.chat.PublicChatConfigStore",
        lambda engine: FakePublicChatConfigStore({"enabled": False}),
    )
    monkeypatch.setattr(
        "app.chat_group_access.ChatRoleAssistantsStore",
        lambda engine: FakeChatRoleAssistantsStore([]),
    )

    response = await chat_endpoint(chat_request, request)
    assert response.status_code == 200
    assert len(store.saved) == 1


@pytest.mark.asyncio
async def test_public_chat_session_denied_when_feature_disabled(monkeypatch):
    """Logged-in public-chat users are subject to the master switch too:
    enabled=false rejects even authenticated chat users."""
    store = FakeMessageStore()
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                rag_application=FakeRAGApplication(),
                app_config=None,
                _message_store=store,
            )
        ),
        state=SimpleNamespace(),
        headers={"content-type": "application/json"},
    )
    chat_request = ChatRequest(
        model="CyberSecurityAssistant",
        messages=[Message(role="user", content="Hallo")],
        stream=False,
    )

    async def fake_get_current_chat_user(request):
        return {
            "id": "chat-user-123",
            "username": "testuser",
            "display_name": "Test User",
            "role": "standard",
            "session_version": 1,
        }

    monkeypatch.setattr(
        "app.routers.chat._get_current_chat_user", fake_get_current_chat_user
    )
    monkeypatch.setattr(
        "app.routers.chat.PublicChatConfigStore",
        lambda engine: FakePublicChatConfigStore({"enabled": False}),
    )
    monkeypatch.setattr(
        "app.chat_group_access.ChatRoleAssistantsStore",
        lambda engine: FakeChatRoleAssistantsStore([]),
    )

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await chat_endpoint(chat_request, request)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_streaming_chat_emits_metrics_delta_chunk(monkeypatch):
    import json

    store = FakeMessageStore()
    usage_meta = {
        "prompt_tokens": 15,
        "completion_tokens": 25,
        "total_tokens": 40,
        "reasoning_tokens": 8,
    }
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                rag_application=FakeRAGApplication(usage_metadata=usage_meta),
                app_config=SimpleNamespace(dashboard_session_secret="test", stream_idle_timeout_seconds=30.0),
                _message_store=store,
            )
        ),
        state=SimpleNamespace(),
        headers={"content-type": "application/json"},
        cookies={},
    )
    chat_request = ChatRequest(
        model="CyberSecurityAssistant",
        messages=[Message(role="user", content="Hallo")],
        stream=True,
    )

    monkeypatch.setattr(
        "app.routers.chat.PublicChatConfigStore",
        lambda engine: FakePublicChatConfigStore(
            {"enabled": True, "enabled_assistants": ["CyberSecurityAssistant"]}
        ),
    )
    monkeypatch.setattr(
        "app.chat_group_access.ChatRoleAssistantsStore",
        lambda engine: FakeChatRoleAssistantsStore(["CyberSecurityAssistant"]),
    )

    response = await chat_endpoint(chat_request, request)

    body = ""
    async for raw in response.body_iterator:
        body += raw

    metrics_chunks = []
    saw_content = False
    for block in body.split("\n\n"):
        if not block.startswith("data: ") or block == "data: [DONE]":
            continue
        payload = json.loads(block[len("data: "):])
        delta = payload["choices"][0].get("delta") or {}
        if delta.get("content"):
            saw_content = True
        if "metrics" in delta:
            metrics_chunks.append(delta["metrics"])

    assert saw_content
    assert len(metrics_chunks) == 1
    m = metrics_chunks[0]
    assert m["prompt_tokens"] == 15
    assert m["completion_tokens"] == 25
    assert m["total_tokens"] == 40
    assert m["reasoning_tokens"] == 8
    assert m["response_time_ms"] is not None and m["response_time_ms"] >= 0
    assert m["tokens_per_second"] is not None and m["tokens_per_second"] > 0


@pytest.mark.asyncio
async def test_streaming_chat_forwards_and_persists_agent_todo_snapshot(monkeypatch):
    import json

    snapshot = {
        "version": 1,
        "tasks": [{"id": "t1", "title": "Read the source", "status": "in_progress"}],
    }

    class TodoRAGApplication(FakeRAGApplication):
        async def run_streaming(self, user_message, *, chat_history, model_name, retrieval_info=None, **kwargs):
            if retrieval_info is not None:
                retrieval_info.update({
                    "status": "skipped",
                    "chunks_retrieved": 0,
                    "rag_context_tokens": 0,
                    "todo_list": snapshot,
                })
            yield {"content": "", "reasoning": "", "todo_list": snapshot}
            yield {"content": "Working on it."}

    store = FakeMessageStore()
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                rag_application=TodoRAGApplication(),
                app_config=SimpleNamespace(dashboard_session_secret="test", stream_idle_timeout_seconds=30.0),
                _message_store=store,
            )
        ),
        state=SimpleNamespace(),
        headers={"content-type": "application/json"},
        cookies={},
    )
    chat_request = ChatRequest(
        model="CyberSecurityAssistant",
        messages=[Message(role="user", content="Handle a multi-step task")],
        stream=True,
    )
    monkeypatch.setattr(
        "app.routers.chat.PublicChatConfigStore",
        lambda engine: FakePublicChatConfigStore(
            {"enabled": True, "enabled_assistants": ["CyberSecurityAssistant"]}
        ),
    )
    monkeypatch.setattr(
        "app.chat_group_access.ChatRoleAssistantsStore",
        lambda engine: FakeChatRoleAssistantsStore(["CyberSecurityAssistant"]),
    )

    response = await chat_endpoint(chat_request, request)
    body = ""
    async for raw in response.body_iterator:
        body += raw

    todo_deltas = []
    for block in body.split("\n\n"):
        if not block.startswith("data: ") or block == "data: [DONE]":
            continue
        payload = json.loads(block[len("data: "):])
        delta = payload["choices"][0].get("delta") or {}
        if "todo_list" in delta:
            todo_deltas.append(delta["todo_list"])

    assert todo_deltas == [snapshot]
    assert len(store.saved) == 1
    assert store.saved[0]["retrieval_info"]["todo_list"] == snapshot


@pytest.fixture(autouse=False)
def clear_run_registry():
    _runs.clear()
    yield
    _runs.clear()


def _streaming_request(rag_app, store, api_key: str | None = "key-1") -> SimpleNamespace:
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                rag_application=rag_app,
                app_config=SimpleNamespace(dashboard_session_secret="test", stream_idle_timeout_seconds=30.0),
                _message_store=store,
            )
        ),
        state=SimpleNamespace(api_key_id=api_key),
        headers={"content-type": "application/json"},
        cookies={},
    )


@pytest.mark.asyncio
async def test_cancel_endpoint_marks_run_and_is_idempotent(clear_run_registry):
    run = ActiveChatRun(
        conversation_id="conv-cancel",
        model_name="CyberSecurityAssistant",
        user_message="hi",
        retrieval_info={},
    )
    register_run(run)
    request = _streaming_request(FakeRAGApplication(), FakeMessageStore())

    result = await chat_run_cancel("conv-cancel", request)
    assert result["cancelled"] is True
    assert run.cancelled and run.cancel_event.is_set()

    result_again = await chat_run_cancel("conv-cancel", request)
    assert result_again["cancelled"] is True

    missing = await chat_run_cancel("conv-missing", request)
    assert missing["cancelled"] is False


@pytest.mark.asyncio
async def test_cancel_endpoint_rejects_unauthorized(clear_run_registry, monkeypatch):
    """Anonymous callers cannot cancel runs of api_accessible=false models."""
    run = ActiveChatRun(
        conversation_id="conv-cancel-403",
        model_name="InternalAssistant",
        user_message="hi",
        retrieval_info={},
    )
    register_run(run)
    monkeypatch.setattr(
        "app.routers.chat.PublicChatConfigStore",
        lambda engine: FakePublicChatConfigStore({"enabled": True, "enabled_assistants": ["InternalAssistant"]}),
    )
    monkeypatch.setattr(
        "app.chat_group_access.ChatRoleAssistantsStore",
        lambda engine: FakeChatRoleAssistantsStore(["InternalAssistant"]),
    )
    request = _streaming_request(FakeRAGApplicationInternalOnly(), FakeMessageStore(), api_key=None)
    with pytest.raises(Exception) as exc_info:
        await chat_run_cancel("conv-cancel-403", request)
    assert getattr(exc_info.value, "status_code", None) == 403
    assert not run.cancelled


@pytest.mark.asyncio
async def test_cancel_endpoint_allows_scoped_public_chat_token(clear_run_registry, monkeypatch):
    """Run endpoints mirror /chat/completions: a scoped token may cancel runs
    of public-chat-enabled internal assistants."""
    run = ActiveChatRun(
        conversation_id="conv-cancel-token",
        model_name="InternalAssistant",
        user_message="hi",
        retrieval_info={},
    )
    register_run(run)
    monkeypatch.setattr(
        "app.routers.chat.PublicChatConfigStore",
        lambda engine: FakePublicChatConfigStore({"enabled": True, "enabled_assistants": ["InternalAssistant"]}),
    )
    monkeypatch.setattr(
        "app.chat_group_access.ChatRoleAssistantsStore",
        lambda engine: FakeChatRoleAssistantsStore(["InternalAssistant"]),
    )
    request = _streaming_request(FakeRAGApplicationInternalOnly(), FakeMessageStore(), api_key=None)
    request.headers["x-public-chat-token"] = _pc_token()
    result = await chat_run_cancel("conv-cancel-token", request)
    assert result["cancelled"] is True
    assert run.cancelled


@pytest.mark.asyncio
async def test_streaming_cancel_stops_producer_gracefully(clear_run_registry, monkeypatch):
    """Cancelling mid-run must close the agent generator (subagent cleanup),
    broadcast a cancelled delta, skip the fallback answer and still persist
    the partial turn."""
    import asyncio
    import json

    class SlowRAGApplication(FakeRAGApplication):
        def __init__(self) -> None:
            super().__init__()
            self.generator_closed = False

        async def run_streaming(self, user_message, *, chat_history, model_name, retrieval_info=None, **kwargs):
            if retrieval_info is not None:
                retrieval_info.update({
                    "status": "ok",
                    "chunks_retrieved": 0,
                    "rag_context_tokens": 0,
                    "rag_context": [],
                })
            try:
                yield {"content": "first-token"}
                while True:
                    await asyncio.sleep(0.02)
                    yield {"content": " tick"}
            finally:
                self.generator_closed = True

    rag_app = SlowRAGApplication()
    store = FakeMessageStore()
    request = _streaming_request(rag_app, store)
    chat_request = ChatRequest(
        model="CyberSecurityAssistant",
        messages=[Message(role="user", content="Hallo")],
        stream=True,
        conversation_id="conv-cancel-prod",
    )
    monkeypatch.setattr(
        "app.routers.chat.PublicChatConfigStore",
        lambda engine: FakePublicChatConfigStore(
            {"enabled": True, "enabled_assistants": ["CyberSecurityAssistant"]}
        ),
    )
    monkeypatch.setattr(
        "app.chat_group_access.ChatRoleAssistantsStore",
        lambda engine: FakeChatRoleAssistantsStore(["CyberSecurityAssistant"]),
    )

    response = await chat_endpoint(chat_request, request)
    run = get_run("conv-cancel-prod")
    assert run is not None

    it = response.body_iterator.__aiter__()
    chunks = [await it.__anext__()]  # role chunk
    # Wait for the first produced token so the generator is inside its try
    # block (aclose() on an unstarted generator would skip the finally).
    async def wait_for_first_token() -> None:
        while True:
            raw = await asyncio.wait_for(it.__anext__(), timeout=10)
            chunks.append(raw)
            if '"first-token"' in raw:
                return
    await wait_for_first_token()
    run.cancel()

    async def drain() -> str:
        parts = list(chunks)
        async for raw in it:
            parts.append(raw)
        return "".join(parts)

    body = await asyncio.wait_for(drain(), timeout=10)

    assert run.cancelled
    assert rag_app.generator_closed  # aclose() reached the generator's finally
    deltas = []
    for block in body.split("\n\n"):
        if not block.startswith("data: ") or block == "data: [DONE]":
            continue
        deltas.append(json.loads(block[len("data: "):])["choices"][0].get("delta") or {})
    assert any(d.get("cancelled") for d in deltas)
    # no fallback answer was synthesised after the cancel
    assert not any(d.get("content") == "No answer found." for d in deltas)
    assert not any("Echo:" in (d.get("content") or "") for d in deltas)
    # partial turn still persisted by the detached producer
    assert len(store.saved) == 1
    assert "first-token" in store.saved[0]["assistant_content"]


@pytest.mark.asyncio
async def test_streaming_chat_metrics_estimated_flag_and_aggregate_tps(monkeypatch):
    """Runs with usage-less iterations surface 'estimated' in the SSE metrics
    trailer, and tokens_per_second is the run's aggregate output rate."""
    import json

    store = FakeMessageStore()
    usage_meta = {
        "prompt_tokens": 15,
        "completion_tokens": 25,
        "total_tokens": 40,
        "tokens_per_second": 21.5,
        "estimated_iterations": 2,
    }
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                rag_application=FakeRAGApplication(usage_metadata=usage_meta),
                app_config=SimpleNamespace(dashboard_session_secret="test", stream_idle_timeout_seconds=30.0),
                _message_store=store,
            )
        ),
        state=SimpleNamespace(),
        headers={"content-type": "application/json"},
        cookies={},
    )
    chat_request = ChatRequest(
        model="CyberSecurityAssistant",
        messages=[Message(role="user", content="Hallo")],
        stream=True,
    )
    monkeypatch.setattr(
        "app.routers.chat.PublicChatConfigStore",
        lambda engine: FakePublicChatConfigStore(
            {"enabled": True, "enabled_assistants": ["CyberSecurityAssistant"]}
        ),
    )
    monkeypatch.setattr(
        "app.chat_group_access.ChatRoleAssistantsStore",
        lambda engine: FakeChatRoleAssistantsStore(["CyberSecurityAssistant"]),
    )

    response = await chat_endpoint(chat_request, request)
    body = ""
    async for raw in response.body_iterator:
        body += raw

    metrics_chunks = []
    for block in body.split("\n\n"):
        if not block.startswith("data: ") or block == "data: [DONE]":
            continue
        payload = json.loads(block[len("data: "):])
        delta = payload["choices"][0].get("delta") or {}
        if "metrics" in delta:
            metrics_chunks.append(delta["metrics"])

    assert len(metrics_chunks) == 1
    m = metrics_chunks[0]
    assert m["tokens_per_second"] == 21.5
    assert m["estimated"] is True


@pytest.mark.asyncio
async def test_streaming_chat_metric_persists_rag_tool_cache_and_tool_tokens(monkeypatch):
    """_save_chat_metric folds rag_search output into rag_context_tokens and
    persists cache_tokens / tool_result_tokens on the row."""
    import json

    store = FakeMessageStore()

    class FakeMetricsStore:
        def __init__(self) -> None:
            self.records = []

        async def save(self, record) -> None:
            self.records.append(record)

    metrics_store = FakeMetricsStore()
    usage_meta = {
        "prompt_tokens": 15,
        "completion_tokens": 25,
        "total_tokens": 40,
        "cache_tokens": 12,
        "tool_result_tokens": 33,
        "rag_tool_tokens": 7,
    }
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                rag_application=FakeRAGApplication(usage_metadata=usage_meta),
                app_config=SimpleNamespace(dashboard_session_secret="test", stream_idle_timeout_seconds=30.0),
                _message_store=store,
                _metrics_store=metrics_store,
            )
        ),
        state=SimpleNamespace(),
        headers={"content-type": "application/json"},
        cookies={},
    )
    chat_request = ChatRequest(
        model="CyberSecurityAssistant",
        messages=[Message(role="user", content="Hallo")],
        stream=True,
    )
    monkeypatch.setattr(
        "app.routers.chat.PublicChatConfigStore",
        lambda engine: FakePublicChatConfigStore(
            {"enabled": True, "enabled_assistants": ["CyberSecurityAssistant"]}
        ),
    )
    monkeypatch.setattr(
        "app.chat_group_access.ChatRoleAssistantsStore",
        lambda engine: FakeChatRoleAssistantsStore(["CyberSecurityAssistant"]),
    )

    response = await chat_endpoint(chat_request, request)
    async for _ in response.body_iterator:
        pass

    assert len(metrics_store.records) == 1
    rec = metrics_store.records[0]
    assert rec.cache_tokens == 12
    assert rec.tool_result_tokens == 33
    # Auto-retrieval (0 in the fake) + rag_search result tokens (7).
    assert rec.rag_context_tokens == 7


def _metric_request(store):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(_metrics_store=store)))


class _CaptureMetricsStore:
    def __init__(self) -> None:
        self.records = []

    async def save(self, record) -> None:
        self.records.append(record)


@pytest.mark.asyncio
async def test_save_chat_metric_tps_fallback_uses_completion_not_total():
    """Fallback must not divide prompt-including total_tokens by time."""
    from app.routers.chat import _save_chat_metric

    store = _CaptureMetricsStore()
    await _save_chat_metric(
        _metric_request(store),
        conversation_id="c1",
        model_name="m",
        message_type="assistant",
        stream=True,
        status_code=200,
        started_at="2025-01-01T00:00:00+00:00",
        response_time_ms=2000.0,
        prompt_tokens=10_000,
        completion_tokens=200,
        total_tokens=10_200,
    )
    rec = store.records[0]
    # 200 completion / 2s = 100 tok/s — NOT 10_200/2 = 5100.
    assert rec.tokens_per_second == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_save_chat_metric_tps_fallback_prefers_llm_generation_ms():
    from app.routers.chat import _save_chat_metric

    store = _CaptureMetricsStore()
    await _save_chat_metric(
        _metric_request(store),
        conversation_id="c1",
        model_name="m",
        message_type="assistant",
        stream=True,
        status_code=200,
        started_at="2025-01-01T00:00:00+00:00",
        response_time_ms=10_000.0,
        llm_generation_ms=1000.0,
        prompt_tokens=5_000,
        completion_tokens=50,
        total_tokens=5_050,
    )
    rec = store.records[0]
    # llm_generation_ms wins over response_time_ms: 50/1s.
    assert rec.tokens_per_second == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_save_chat_metric_tps_explicit_wins():
    from app.routers.chat import _save_chat_metric

    store = _CaptureMetricsStore()
    await _save_chat_metric(
        _metric_request(store),
        conversation_id="c1",
        model_name="m",
        message_type="assistant",
        stream=True,
        status_code=200,
        started_at="2025-01-01T00:00:00+00:00",
        response_time_ms=2000.0,
        prompt_tokens=10_000,
        completion_tokens=200,
        total_tokens=10_200,
        tokens_per_second=42.5,
    )
    assert store.records[0].tokens_per_second == 42.5


@pytest.mark.asyncio
async def test_save_chat_metric_tps_missing_inputs_give_none():
    from app.routers.chat import _save_chat_metric

    store = _CaptureMetricsStore()
    for kwargs in (
        # No completion tokens
        {"completion_tokens": None, "total_tokens": 1000},
        # No usable duration
        {"completion_tokens": 100, "total_tokens": 1000, "response_time_ms": 0.0},
    ):
        base = dict(
            conversation_id="c1",
            model_name="m",
            message_type="assistant",
            stream=False,
            status_code=200,
            started_at="2025-01-01T00:00:00+00:00",
            response_time_ms=kwargs.pop("response_time_ms", 2000.0),
            prompt_tokens=500,
        )
        await _save_chat_metric(_metric_request(store), **{**base, **kwargs})
    assert store.records[0].tokens_per_second is None
    assert store.records[1].tokens_per_second is None
