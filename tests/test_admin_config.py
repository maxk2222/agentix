"""Tests for the admin configuration and knowledge database endpoints."""

from __future__ import annotations

import os
import threading
import time
from types import SimpleNamespace
from pathlib import Path

import pytest
import yaml

from app.routers.admin import (
    get_instructions,
    put_instructions,
    config_apply,
    knowledge_collections,
    knowledge_collection_stats,
    knowledge_preview,
    knowledge_upload,
    knowledge_delete,
    knowledge_create_collection,
    knowledge_delete_collection,
    InstructionsPayload,
    AssistantEntry,
)


def _make_request(app_config, **extra_state):
    state = SimpleNamespace(app_config=app_config, **extra_state)
    request = SimpleNamespace(app=SimpleNamespace(state=state), cookies={})
    return request


@pytest.fixture(autouse=True)
def patch_auth(monkeypatch):
    """Bypass admin auth for all tests in this module."""
    monkeypatch.setattr(
        "app.routers.admin.require_admin", lambda request: None
    )
    async def _noop_permission(request, permission):
        pass
    monkeypatch.setattr(
        "app.routers.admin.require_permission", _noop_permission
    )


def _build_app_config(tmp_path: Path):
    import os
    os.environ["DATA_DIR"] = str(tmp_path / "data")
    os.environ["INSTRUCTIONS_FILE"] = str(tmp_path / "instructions.yaml")
    from agent.config import AppConfig
    instructions_file = tmp_path / "instructions.yaml"
    instructions_file.write_text(
        yaml.safe_dump({
            "assistants": [
                {
                    "name": "TestBot",
                    "model": "gpt-4",
                    "temperature": 0.5,
                    "system_instructions": "Be helpful.",
                }
            ]
        }),
        encoding="utf-8",
    )
    return AppConfig(
        pg_host="localhost",
        pg_port=5432,
        pg_database="test_db",
        pg_user="test",
        pg_password="test",
        dashboard_enabled=True,
        dashboard_admin_password="secret",
        dashboard_session_secret="test-secret-123",
    )


@pytest.mark.asyncio
async def test_get_instructions(tmp_path: Path):
    app_config = _build_app_config(tmp_path)
    request = _make_request(app_config)
    result = await get_instructions(request)
    assert "assistants" in result
    assert len(result["assistants"]) == 1
    assert result["assistants"][0]["name"] == "TestBot"


@pytest.mark.asyncio
async def test_get_instructions_missing_file(tmp_path: Path):
    app_config = _build_app_config(tmp_path)
    app_config.instructions_file = Path("/nonexistent/instructions.yaml")
    request = _make_request(app_config)
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        await get_instructions(request)
    assert exc_info.value.status_code == 404


def test_bootstrap_default_runtime_data_copies_default_bundle_when_primary_missing(tmp_path: Path, monkeypatch):
    app_config = _build_app_config(tmp_path)
    fallback_root = tmp_path / "defaults"
    fallback_file = fallback_root / "SystemInstructions" / "instructions.yaml"
    fallback_file.parent.mkdir(parents=True)
    (fallback_root / "KnowledgeDatabase" / "BSI").mkdir(parents=True)
    (fallback_root / "KnowledgeDatabase" / "BSI" / "sample.txt").write_text("sample", encoding="utf-8")
    fallback_file.write_text(
        yaml.safe_dump(
            {
                "assistants": [
                    {
                        "name": "FallbackBot",
                        "model": "gpt-4o-mini",
                        "temperature": 0.0,
                        "system_instructions": "Fallback.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("agent.config.REPO_DEFAULTS_ROOT", fallback_root)
    monkeypatch.setattr("agent.config.PACKAGED_DEFAULTS_ROOT", tmp_path / "unused-packaged-defaults")
    app_config.instructions_file = tmp_path / "data" / "SystemInstructions" / "instructions.yaml"

    copied = app_config.bootstrap_default_runtime_data()

    assert copied is True
    assert app_config.instructions_file.exists()
    assert yaml.safe_load(app_config.instructions_file.read_text(encoding="utf-8"))["assistants"][0]["name"] == "FallbackBot"
    assert (app_config.data_dir / "KnowledgeDatabase" / "BSI" / "sample.txt").exists()
    assert (app_config.data_dir / "SystemInstructions" / "instructions.yaml").exists()


def test_bootstrap_default_runtime_data_keeps_primary_paths_when_instructions_exist(tmp_path: Path):
    app_config = _build_app_config(tmp_path)
    primary_file = app_config.instructions_file
    app_config.data_dir.mkdir(parents=True)

    copied = app_config.bootstrap_default_runtime_data()

    assert copied is False
    assert app_config.instructions_file == primary_file
    assert app_config.data_dir.exists()


@pytest.mark.asyncio
async def test_put_instructions(tmp_path: Path):
    app_config = _build_app_config(tmp_path)
    request = _make_request(app_config)
    payload = InstructionsPayload(
        assistants=[
            AssistantEntry(
                name="UpdatedBot",
                model="gpt-4o",
                system_instructions="Be very helpful.",
                temperature=0.7,
            )
        ]
    )
    result = await put_instructions(request, payload)
    assert result["success"] is True

    # Verify file was rewritten
    data = yaml.safe_load(app_config.instructions_file.read_text(encoding="utf-8"))
    assert len(data["assistants"]) == 1
    assert data["assistants"][0]["name"] == "UpdatedBot"
    assert data["assistants"][0]["temperature"] == 0.7


@pytest.mark.asyncio
async def test_put_instructions_extra_body_roundtrip(tmp_path: Path):
    app_config = _build_app_config(tmp_path)
    request = _make_request(app_config)
    payload = InstructionsPayload(
        assistants=[
            AssistantEntry(
                name="ExtraBot",
                model="gpt-4o",
                system_instructions="Be helpful.",
                temperature=0.5,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                extra_parameters={"top_p": 0.9},
                enable_thinking=False,
            )
        ]
    )
    result = await put_instructions(request, payload)
    assert result["success"] is True

    data = yaml.safe_load(app_config.instructions_file.read_text(encoding="utf-8"))
    assert data["assistants"][0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert data["assistants"][0]["extra_parameters"] == {"top_p": 0.9}
    assert data["assistants"][0]["enable_thinking"] is False

    # The runtime loader must preserve the fields (regression: they were
    # dropped by the field whitelist and never reached the LLM).
    from app.chat_utils import load_instructions_from_yaml_text
    parsed = load_instructions_from_yaml_text(
        app_config.instructions_file.read_text(encoding="utf-8")
    )
    assert parsed["ExtraBot"]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert parsed["ExtraBot"]["extra_parameters"] == {"top_p": 0.9}
    assert parsed["ExtraBot"]["enable_thinking"] is False


@pytest.mark.asyncio
async def test_put_instructions_workspace_roundtrip(tmp_path: Path):
    """workspace + subagents.exclude_tools must survive save AND the runtime
    loader (regression: both were dropped by field whitelists)."""
    app_config = _build_app_config(tmp_path)
    request = _make_request(app_config)
    payload = InstructionsPayload(
        assistants=[
            AssistantEntry(
                name="WsBot",
                model="gpt-4o",
                system_instructions="Be helpful.",
                agent_enabled=True,
                workspace={
                    "enabled": True,
                    "scope": "user",
                    "api_enabled": True,
                    "vision_read": True,
                    "max_file_mb": 10,
                },
                subagents={
                    "enabled": True,
                    "exclude_tools": ["code_execution"],
                    "report_answer_chars": 5000,
                },
            )
        ]
    )
    result = await put_instructions(request, payload)
    assert result["success"] is True

    data = yaml.safe_load(app_config.instructions_file.read_text(encoding="utf-8"))
    ws = data["assistants"][0]["workspace"]
    assert ws["enabled"] is True
    assert ws["scope"] == "user"
    assert ws["api_enabled"] is True
    assert ws["vision_read"] is True
    assert ws["max_file_mb"] == 10

    from app.chat_utils import load_instructions_from_yaml_text
    parsed = load_instructions_from_yaml_text(
        app_config.instructions_file.read_text(encoding="utf-8")
    )
    assert parsed["WsBot"]["workspace"]["scope"] == "user"
    assert parsed["WsBot"]["workspace"]["api_enabled"] is True
    assert parsed["WsBot"]["workspace"]["vision_read"] is True
    assert parsed["WsBot"]["subagents"]["exclude_tools"] == ["code_execution"]
    assert parsed["WsBot"]["subagents"]["report_answer_chars"] == 5000


@pytest.mark.asyncio
async def test_put_instructions_workspace_disabled_drops_key(tmp_path: Path):
    app_config = _build_app_config(tmp_path)
    request = _make_request(app_config)
    payload = InstructionsPayload(
        assistants=[
            AssistantEntry(
                name="NoWs",
                model="gpt-4o",
                system_instructions="Be helpful.",
                workspace={"enabled": False},
            )
        ]
    )
    await put_instructions(request, payload)
    data = yaml.safe_load(app_config.instructions_file.read_text(encoding="utf-8"))
    assert "workspace" not in data["assistants"][0]


@pytest.mark.asyncio
async def test_put_instructions_workspace_api_requires_enabled(tmp_path: Path):
    app_config = _build_app_config(tmp_path)
    request = _make_request(app_config)
    payload = InstructionsPayload(
        assistants=[
            AssistantEntry(
                name="BadWs",
                model="gpt-4o",
                system_instructions="Be helpful.",
                workspace={"enabled": False, "api_enabled": True},
            )
        ]
    )
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        await put_instructions(request, payload)
    assert exc_info.value.status_code == 400
    assert "api_enabled requires" in exc_info.value.detail


@pytest.mark.asyncio
async def test_put_instructions_rename_moves_workspace(tmp_path: Path):
    """Renaming an assistant in place moves its workspace tree."""
    app_config = _build_app_config(tmp_path)
    request = _make_request(app_config)

    from app.agent_workspace import AgentWorkspaceStore
    store = AgentWorkspaceStore(tmp_path / "data" / "agent_workspaces")
    store.write(store.scope_dir("TestBot", "assistant"), "notes.txt", b"keep")

    payload = InstructionsPayload(
        assistants=[
            AssistantEntry(
                name="RenamedBot",
                model="gpt-4",
                system_instructions="sys",
            )
        ]
    )
    await put_instructions(request, payload)

    assert not (tmp_path / "data" / "agent_workspaces" / "TestBot").exists()
    moved = tmp_path / "data" / "agent_workspaces" / "RenamedBot" / "shared" / "notes.txt"
    assert moved.read_bytes() == b"keep"


@pytest.mark.asyncio
async def test_put_instructions_delete_plus_add_does_not_rename(tmp_path: Path):
    """Delete A + append B must NOT leak A's workspace into B (position differs)."""
    app_config = _build_app_config(tmp_path)
    # Seed two assistants so removal/addition land on different positions.
    app_config.instructions_file.write_text(
        yaml.safe_dump({
            "assistants": [
                {"name": "TestBot", "model": "gpt-4", "system_instructions": "a"},
                {"name": "KeepBot", "model": "gpt-4", "system_instructions": "b"},
            ]
        }),
        encoding="utf-8",
    )
    request = _make_request(app_config)

    from app.agent_workspace import AgentWorkspaceStore
    store = AgentWorkspaceStore(tmp_path / "data" / "agent_workspaces")
    store.write(store.scope_dir("TestBot", "assistant"), "notes.txt", b"keep")

    payload = InstructionsPayload(
        assistants=[
            AssistantEntry(name="KeepBot", model="gpt-4", system_instructions="b"),
            AssistantEntry(name="NewBot", model="gpt-4", system_instructions="c"),
        ]
    )
    await put_instructions(request, payload)

    # TestBot (index 0) removed, NewBot added at index 1 — not a same-slot
    # rename, so TestBot's tree must stay under its own (orphaned) name.
    assert (tmp_path / "data" / "agent_workspaces" / "TestBot" / "shared" / "notes.txt").exists()
    assert not (tmp_path / "data" / "agent_workspaces" / "NewBot").exists()


@pytest.mark.asyncio
async def test_put_instructions_validation_error(tmp_path: Path):
    app_config = _build_app_config(tmp_path)
    request = _make_request(app_config)
    payload = InstructionsPayload(
        assistants=[AssistantEntry(name="", model="", system_instructions="")]
    )
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        await put_instructions(request, payload)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_config_apply(tmp_path: Path, monkeypatch):
    """config_apply should touch main.py so uvicorn reload restarts."""
    app_config = _build_app_config(tmp_path)
    class FakeRAG:
        async def force_refresh_instructions(self):
            return True
    request = _make_request(app_config, rag_application=FakeRAG())
    touched: list[Path] = []

    def fake_touch(path: Path):
        touched.append(path)

    def fake_thread_start(self):
        # Run target synchronously for test determinism
        if self._target:
            self._target(*self._args, **self._kwargs)

    monkeypatch.setattr(Path, "touch", fake_touch)
    monkeypatch.setattr(threading.Thread, "start", fake_thread_start)
    monkeypatch.setattr(time, "sleep", lambda x: None)

    result = await config_apply(request)
    assert "Configuration initialized." in result["message"]
    assert result["reloaded"] is True


@pytest.mark.asyncio
async def test_knowledge_collections(tmp_path: Path):
    app_config = _build_app_config(tmp_path)
    kb = tmp_path / "data" / "KnowledgeDatabase"
    (kb / "BSI").mkdir(parents=True)
    (kb / "etc").mkdir(parents=True)
    (kb / "BSI" / "doc1.pdf").write_text("pdf")
    (kb / "etc" / "doc2.txt").write_text("txt")

    request = _make_request(app_config)
    result = await knowledge_collections(request)
    names = {c.name for c in result}
    assert names == {"BSI", "etc"}
    bsi = next(c for c in result if c.name == "BSI")
    assert bsi.files == ["doc1.pdf"]


@pytest.mark.asyncio
async def test_knowledge_upload_and_delete(tmp_path: Path):
    app_config = _build_app_config(tmp_path)
    kb = tmp_path / "data" / "KnowledgeDatabase"
    request = _make_request(app_config)

    class FakeUploadFile:
        filename = "test.pdf"
        async def read(self):
            return b"PDF content"

    class FakeUploadFileTwo:
        filename = "notes.txt"
        async def read(self):
            return b"notes"

    result = await knowledge_upload(request, collection="BSI", file=[FakeUploadFile(), FakeUploadFileTwo()])
    assert result["success"] is True
    assert sorted(result["filenames"]) == ["notes.txt", "test.pdf"]
    assert (kb / "BSI" / "test.pdf").exists()
    assert (kb / "BSI" / "notes.txt").exists()

    # Delete
    del_result = await knowledge_delete(request, collection="BSI", filename="test.pdf")
    assert del_result["success"] is True
    assert not (kb / "BSI" / "test.pdf").exists()
    assert (kb / "BSI" / "notes.txt").exists()


@pytest.mark.asyncio
async def test_knowledge_upload_invalid_extension(tmp_path: Path):
    app_config = _build_app_config(tmp_path)
    request = _make_request(app_config)

    class FakeUploadFile:
        filename = "virus.exe"
        async def read(self):
            return b"bad"

    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        await knowledge_upload(request, collection="BSI", file=[FakeUploadFile()])
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_knowledge_upload_invalid_collection(tmp_path: Path):
    app_config = _build_app_config(tmp_path)
    request = _make_request(app_config)

    class FakeUploadFile:
        filename = "test.pdf"
        async def read(self):
            return b"pdf"

    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        await knowledge_upload(request, collection="../etc", file=[FakeUploadFile()])
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_knowledge_preview_pdf_returns_pdf(tmp_path: Path):
    app_config = _build_app_config(tmp_path)
    kb = tmp_path / "data" / "KnowledgeDatabase"
    pdf = kb / "BSI" / "sample.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.4\n%fake")
    request = _make_request(app_config)

    response = await knowledge_preview(request, collection="BSI", filename="sample.pdf")

    assert response.media_type == "application/pdf"


@pytest.mark.asyncio
async def test_knowledge_delete_not_found(tmp_path: Path):
    app_config = _build_app_config(tmp_path)
    request = _make_request(app_config)
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        await knowledge_delete(request, collection="missing", filename="nope.pdf")
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_knowledge_create_and_delete_collection(tmp_path: Path):
    app_config = _build_app_config(tmp_path)
    kb = tmp_path / "data" / "KnowledgeDatabase"
    request = _make_request(app_config)

    result = await knowledge_create_collection(request, name="NewCol")
    assert result["success"] is True
    assert (kb / "NewCol").exists()

    del_result = await knowledge_delete_collection(request, name="NewCol")
    assert del_result["success"] is True
    assert not (kb / "NewCol").exists()


@pytest.mark.asyncio
async def test_knowledge_collections_with_subfolders(tmp_path: Path):
    app_config = _build_app_config(tmp_path)
    kb = tmp_path / "data" / "KnowledgeDatabase"
    (kb / "BSI" / "2024").mkdir(parents=True)
    (kb / "BSI" / "2024" / "report.pdf").write_text("pdf")
    (kb / "BSI" / "doc1.pdf").write_text("pdf")

    request = _make_request(app_config)
    result = await knowledge_collections(request)
    bsi = next(c for c in result if c.name == "BSI")
    assert bsi.files == ["doc1.pdf"]
    assert len(bsi.subfolders) == 1
    assert bsi.subfolders[0].name == "2024"
    assert bsi.subfolders[0].files == ["report.pdf"]


@pytest.mark.asyncio
async def test_knowledge_collections_include_counts_false_by_default(tmp_path: Path):
    app_config = _build_app_config(tmp_path)
    kb = tmp_path / "data" / "KnowledgeDatabase"
    (kb / "BSI").mkdir(parents=True)
    (kb / "BSI" / "doc1.pdf").write_text("pdf")

    request = _make_request(app_config)
    result = await knowledge_collections(request)
    bsi = next(c for c in result if c.name == "BSI")
    assert bsi.vector_doc_count == 0
    assert bsi.file_vector_doc_count == {}


@pytest.mark.asyncio
async def test_knowledge_collections_include_counts_true(tmp_path: Path):
    from unittest.mock import AsyncMock, MagicMock

    app_config = _build_app_config(tmp_path)
    kb = tmp_path / "data" / "KnowledgeDatabase"
    (kb / "BSI").mkdir(parents=True)
    (kb / "BSI" / "doc1.pdf").write_text("pdf")

    fake_manager = MagicMock()
    fake_manager.get_document_count = AsyncMock(return_value=42)
    fake_manager.count_documents_for_sources = AsyncMock(return_value={"BSI/doc1.pdf": 7})

    rag_app = MagicMock()
    rag_app.vector_managers = {"BSI": fake_manager}
    rag_app.engine = None

    request = _make_request(app_config, rag_application=rag_app)
    result = await knowledge_collections(request, include_counts=True)
    bsi = next(c for c in result if c.name == "BSI")
    assert bsi.vector_doc_count == 42
    fake_manager.get_document_count.assert_awaited_once()
    fake_manager.count_documents_for_sources.assert_not_awaited()  # no DB sources for this collection


@pytest.mark.asyncio
async def test_knowledge_collection_stats(tmp_path: Path):
    from unittest.mock import AsyncMock, MagicMock, patch

    app_config = _build_app_config(tmp_path)
    kb = tmp_path / "data" / "KnowledgeDatabase"
    (kb / "BSI").mkdir(parents=True)

    fake_manager = MagicMock()
    fake_manager.get_document_count = AsyncMock(return_value=99)
    fake_manager.count_documents_for_sources = AsyncMock(return_value={"BSI/doc1.pdf": 5})

    rag_app = MagicMock()
    rag_app.vector_managers = {"BSI": fake_manager}

    # Mock the engine and source state query
    fake_conn = MagicMock()
    fake_result = MagicMock()
    fake_result.__iter__ = lambda self: iter([MagicMock(source="BSI/doc1.pdf")])
    fake_conn.execute = AsyncMock(return_value=fake_result)
    fake_conn.__aenter__ = AsyncMock(return_value=fake_conn)
    fake_conn.__aexit__ = AsyncMock(return_value=None)

    fake_engine = MagicMock()
    fake_engine.connect = MagicMock(return_value=fake_conn)
    rag_app.engine = fake_engine

    request = _make_request(app_config, rag_application=rag_app)
    result = await knowledge_collection_stats(request, name="BSI")
    assert result["vector_doc_count"] == 99
    assert result["file_vector_doc_count"] == {"doc1.pdf": 5}
    fake_manager.count_documents_for_sources.assert_awaited_once_with(["BSI/doc1.pdf"])


@pytest.mark.asyncio
async def test_knowledge_collection_stats_no_manager(tmp_path: Path):
    app_config = _build_app_config(tmp_path)
    request = _make_request(app_config, rag_application=None)
    result = await knowledge_collection_stats(request, name="missing")
    assert result["vector_doc_count"] == 0
    assert result["file_vector_doc_count"] == {}


@pytest.mark.asyncio
async def test_reasoning_schema_endpoint_known_model(tmp_path: Path):
    from app.routers.admin import reasoning_schema

    app_config = _build_app_config(tmp_path)
    request = _make_request(app_config)
    result = await reasoning_schema(request, model="Qwen3-8B")
    assert result["family"] == "qwen_thinking"
    assert result["assumed"] is False
    keys = {c["key"] for c in result["controls"]}
    assert keys == {"enabled", "budget"}


@pytest.mark.asyncio
async def test_reasoning_schema_endpoint_unknown_model(tmp_path: Path):
    from app.routers.admin import reasoning_schema

    app_config = _build_app_config(tmp_path)
    request = _make_request(app_config)
    result = await reasoning_schema(request, model="some-custom-llm")
    assert result["family"] == "generic"
    assert result["assumed"] is True


@pytest.mark.asyncio
async def test_put_instructions_reasoning_normalized(tmp_path: Path):
    """Reasoning dicts are normalized to canonical keys on save."""
    app_config = _build_app_config(tmp_path)
    request = _make_request(app_config)
    payload = InstructionsPayload(
        assistants=[
            AssistantEntry(
                name="TestBot",
                model="Qwen3-8B",
                temperature=0.5,
                system_instructions="Be helpful.",
                reasoning={"enabled": False, "budget": 1024, "bogus_key": "x"},
            )
        ]
    )
    await put_instructions(request, payload)
    instructions_file = tmp_path / "instructions.yaml"
    data = yaml.safe_load(instructions_file.read_text())
    assert data["assistants"][0]["reasoning"] == {"enabled": False, "budget": 1024}


@pytest.mark.asyncio
async def test_put_instructions_reasoning_invalid_rejected(tmp_path: Path):
    from fastapi import HTTPException

    app_config = _build_app_config(tmp_path)
    request = _make_request(app_config)
    payload = InstructionsPayload(
        assistants=[
            AssistantEntry(
                name="TestBot",
                model="gpt-4",
                temperature=0.5,
                system_instructions="Be helpful.",
                reasoning={"totally_unknown": "x"},
            )
        ]
    )
    with pytest.raises(HTTPException) as exc_info:
        await put_instructions(request, payload)
    assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# Provider admin mutations → background membership refresh
# ---------------------------------------------------------------------------


class _FakeProviderStore:
    def __init__(self, providers=None):
        self._providers = dict(providers or {})
        self.created = []
        self.updated = []
        self.deleted = []
        self.rotated = []

    async def get_provider(self, provider_id):
        return self._providers.get(provider_id)

    async def create_provider(self, provider):
        self._providers[provider.id] = provider
        self.created.append(provider.id)

    async def update_provider(self, provider):
        self._providers[provider.id] = provider
        self.updated.append(provider.id)

    async def delete_provider(self, provider_id):
        self._providers.pop(provider_id, None)
        self.deleted.append(provider_id)

    async def rotate_api_key(self, provider_id, key):
        self.rotated.append((provider_id, key))


class _FakeRagScheduler:
    def __init__(self):
        self.refresh_scheduled = 0

    def schedule_provider_model_refresh(self):
        self.refresh_scheduled += 1


def _provider_request(app_config, store, rag=None):
    return _make_request(app_config, _provider_store=store, rag_application=rag)


def _provider_payload(**overrides):
    from app.routers.admin import ProviderPayload
    data = dict(id="p1", name="P1", base_url="http://host/v1")
    data.update(overrides)
    return ProviderPayload(**data)


@pytest.mark.asyncio
async def test_create_provider_schedules_refresh_and_normalizes_url(tmp_path):
    from app.routers.admin import create_provider
    app_config = _build_app_config(tmp_path)
    store, rag = _FakeProviderStore(), _FakeRagScheduler()
    request = _provider_request(app_config, store, rag)

    await create_provider(request, _provider_payload())
    assert store.created == ["p1"]
    assert store._providers["p1"].base_url == "http://host"  # /v1 normalized on write
    assert rag.refresh_scheduled == 1


@pytest.mark.asyncio
async def test_update_provider_schedules_refresh(tmp_path):
    from app.routers.admin import update_provider
    app_config = _build_app_config(tmp_path)
    existing = SimpleNamespace(id="p1")
    store, rag = _FakeProviderStore({"p1": existing}), _FakeRagScheduler()
    request = _provider_request(app_config, store, rag)

    await update_provider(request, "p1", _provider_payload(base_url="http://h2"))
    assert store.updated == ["p1"]
    assert rag.refresh_scheduled == 1


@pytest.mark.asyncio
async def test_delete_provider_schedules_refresh(tmp_path):
    from app.routers.admin import delete_provider
    app_config = _build_app_config(tmp_path)
    store, rag = _FakeProviderStore({"p1": SimpleNamespace(id="p1")}), _FakeRagScheduler()
    request = _provider_request(app_config, store, rag)

    await delete_provider(request, "p1")
    assert store.deleted == ["p1"]
    assert rag.refresh_scheduled == 1


@pytest.mark.asyncio
async def test_rotate_secret_schedules_refresh(tmp_path):
    from app.routers.admin import rotate_provider_secret, SecretRotatePayload
    app_config = _build_app_config(tmp_path)
    store, rag = _FakeProviderStore({"p1": SimpleNamespace(id="p1")}), _FakeRagScheduler()
    request = _provider_request(app_config, store, rag)

    await rotate_provider_secret(request, "p1", SecretRotatePayload(api_key="new-key"))
    assert store.rotated == [("p1", "new-key")]
    assert rag.refresh_scheduled == 1


@pytest.mark.asyncio
async def test_import_providers_schedules_refresh(tmp_path):
    from app.routers.admin import import_providers
    app_config = _build_app_config(tmp_path)
    store, rag = _FakeProviderStore(), _FakeRagScheduler()
    request = _provider_request(app_config, store, rag)

    await import_providers(request, _provider_payload())
    assert store.created == ["p1"]
    assert rag.refresh_scheduled == 1


@pytest.mark.asyncio
async def test_schedule_provider_model_refresh_is_nonblocking(tmp_path):
    """schedule() returns immediately and coalesces into one worker task."""
    import asyncio
    from agent.application import RAGApplication

    class _Store:
        def __init__(self):
            self.list_calls = 0
        async def get_enabled_providers(self):
            self.list_calls += 1
            return []
        async def fetch_models(self, pid):
            return []

    app = RAGApplication.__new__(RAGApplication)
    app._provider_store = _Store()
    app._provider_model_cache = {}
    app._provider_cache_lock = asyncio.Lock()
    app._provider_refresh_task = None
    app._provider_refresh_requested = False
    app._chain_builder = None

    app.schedule_provider_model_refresh()
    app.schedule_provider_model_refresh()  # coalesced, no second task
    task = app._provider_refresh_task
    assert task is not None
    await task
    assert app._provider_store.list_calls >= 1


# ---------------------------------------------------------------------------
# Serving-provider resolution (mirrors ChainBuilder._resolve_provider_endpoint)
# ---------------------------------------------------------------------------


class _Prov:
    def __init__(self, pid, enabled=True, native=None):
        self.id = pid
        self.enabled = enabled
        self.supports_native_tool_calls = native


class _ServingStore:
    def __init__(self, providers=(), default_id=None, hints=None):
        self._providers = {p.id: p for p in providers}
        self._default_id = default_id
        self._hints = hints or {}

    async def get_provider(self, pid):
        return self._providers.get(pid)

    async def get_default_provider(self):
        return self._providers.get(self._default_id)

    async def get_model_tool_calling_status(self, pid, model):
        return self._hints.get((pid, model))


def _make_app_for_serving(store, instructions, membership=None):
    from agent.application import RAGApplication

    app = RAGApplication.__new__(RAGApplication)
    app._provider_store = store
    app._provider_model_cache = membership or {}
    app.instructions = instructions
    return app


@pytest.mark.asyncio
async def test_serving_provider_explicit_pin_wins():
    store = _ServingStore(
        [_Prov("pin"), _Prov("mapped"), _Prov("dflt")], default_id="dflt"
    )
    app = _make_app_for_serving(
        store,
        {"A": {"model": "m1", "provider_id": "pin"}},
        membership={"m1": "mapped"},
    )
    provider, model = await app._resolve_serving_provider("A")
    assert provider.id == "pin"
    assert model == "m1"


@pytest.mark.asyncio
async def test_serving_provider_disabled_pin_falls_to_membership():
    store = _ServingStore(
        [_Prov("pin", enabled=False), _Prov("mapped")], default_id=None
    )
    app = _make_app_for_serving(
        store,
        {"A": {"model": "m1", "provider_id": "pin"}},
        membership={"m1": "mapped"},
    )
    provider, model = await app._resolve_serving_provider("A")
    assert provider.id == "mapped"
    assert model == "m1"


@pytest.mark.asyncio
async def test_serving_provider_disabled_pin_and_map_falls_to_default():
    store = _ServingStore(
        [_Prov("pin", enabled=False), _Prov("mapped", enabled=False), _Prov("dflt")],
        default_id="dflt",
    )
    app = _make_app_for_serving(
        store,
        {"A": {"model": "m1", "provider_id": "pin"}},
        membership={"m1": "mapped"},
    )
    provider, model = await app._resolve_serving_provider("A")
    assert provider.id == "dflt"
    assert model == "m1"


@pytest.mark.asyncio
async def test_serving_provider_disabled_default_returns_none():
    store = _ServingStore([_Prov("dflt", enabled=False)], default_id="dflt")
    app = _make_app_for_serving(store, {"A": {"model": "m1"}})
    provider, _ = await app._resolve_serving_provider("A")
    assert provider is None


@pytest.mark.asyncio
async def test_tool_calling_hint_uses_serving_provider_not_dead_pin():
    """Hint must describe the provider that actually serves the model."""
    store = _ServingStore(
        [_Prov("pin", enabled=False), _Prov("dflt")],
        default_id="dflt",
        hints={("dflt", "m1"): True},
    )
    app = _make_app_for_serving(
        store, {"A": {"model": "m1", "provider_id": "pin"}}
    )
    assert await app._get_provider_tool_calling_hint("A") is True


@pytest.mark.asyncio
async def test_tool_calling_hint_provider_wide_fallback():
    store = _ServingStore([_Prov("dflt", native=False)], default_id="dflt")
    app = _make_app_for_serving(store, {"A": {"model": "m1"}})
    assert await app._get_provider_tool_calling_hint("A") is False
