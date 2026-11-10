from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.api_keys import AppApiKey, AppApiKeyStore


class FakeAsyncResult:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return FakeMappings(self._rows)


class FakeMappings:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class FakeConn:
    def __init__(self):
        self.executed = []
        self._rows = []

    async def execute(self, query, params=None):
        self.executed.append((query, params))
        return FakeAsyncResult(self._rows)


class FakeEngine:
    def __init__(self):
        self._conn = FakeConn()

    def begin(self):
        return FakeTransaction(self._conn)


class FakeTransaction:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def store():
    return AppApiKeyStore(FakeEngine())


@pytest.mark.asyncio
async def test_ensure_tables(store):
    await store.ensure_tables()
    assert len(store._engine._conn.executed) == 2
    assert "CREATE TABLE IF NOT EXISTS app_api_keys" in str(store._engine._conn.executed[0][0])
    assert "ADD COLUMN IF NOT EXISTS rate_limit_rpm" in str(store._engine._conn.executed[1][0])


@pytest.mark.asyncio
async def test_generate_and_hash_key(store):
    key = store.generate_key()
    assert key.startswith("ak_")
    hashed = store.hash_key(key)
    assert len(hashed) == 64  # SHA-256 hex


@pytest.mark.asyncio
async def test_create_and_list_key(store):
    await store.ensure_tables()
    key = AppApiKey(
        id="test-1",
        name="Test Key",
        api_key_hash=store.hash_key("secret123"),
        allowed_models=["model-a", "model-b"],
        enabled=True,
    )
    await store.create_key(key)

    # Mock list response
    store._engine._conn.executed = []
    store._engine._conn._rows = [
        {
            "id": "test-1",
            "name": "Test Key",
            "api_key_hash": store.hash_key("secret123"),
            "allowed_models": ["model-a", "model-b"],
            "enabled": True,
            "created_at": None,
            "updated_at": None,
        }
    ]

    keys = await store.list_keys()
    assert len(keys) == 1
    assert keys[0].id == "test-1"
    assert keys[0].allowed_models == ["model-a", "model-b"]


@pytest.mark.asyncio
async def test_verify_key(store):
    await store.ensure_tables()
    raw_key = store.generate_key()
    key = AppApiKey(
        id="test-2",
        name="Verify Key",
        api_key_hash=store.hash_key(raw_key),
        allowed_models=[],
        enabled=True,
    )
    await store.create_key(key)

    # Mock list for verify
    store._engine._conn.executed = []
    store._engine._conn._rows = [
        {
            "id": "test-2",
            "name": "Verify Key",
            "api_key_hash": store.hash_key(raw_key),
            "allowed_models": [],
            "enabled": True,
            "created_at": None,
            "updated_at": None,
        }
    ]

    result = await store.verify_key(raw_key)
    assert result is not None
    assert result.id == "test-2"

    result_bad = await store.verify_key("wrong-key")
    assert result_bad is None


@pytest.mark.asyncio
async def test_disabled_key_not_verified(store):
    await store.ensure_tables()
    raw_key = store.generate_key()
    key = AppApiKey(
        id="test-3",
        name="Disabled Key",
        api_key_hash=store.hash_key(raw_key),
        allowed_models=[],
        enabled=False,
    )
    await store.create_key(key)

    store._engine._conn.executed = []
    store._engine._conn._rows = [
        {
            "id": "test-3",
            "name": "Disabled Key",
            "api_key_hash": store.hash_key(raw_key),
            "allowed_models": [],
            "enabled": False,
            "created_at": None,
            "updated_at": None,
        }
    ]

    result = await store.verify_key(raw_key)
    assert result is None


@pytest.mark.asyncio
async def test_is_model_allowed(store):
    await store.ensure_tables()
    raw_key = store.generate_key()
    key = AppApiKey(
        id="test-4",
        name="Model Key",
        api_key_hash=store.hash_key(raw_key),
        allowed_models=["model-a"],
        enabled=True,
    )
    await store.create_key(key)

    store._engine._conn.executed = []
    store._engine._conn._rows = [
        {
            "id": "test-4",
            "name": "Model Key",
            "api_key_hash": store.hash_key(raw_key),
            "allowed_models": ["model-a"],
            "enabled": True,
            "created_at": None,
            "updated_at": None,
        }
    ]

    assert await store.is_model_allowed(raw_key, "model-a") is True
    assert await store.is_model_allowed(raw_key, "model-b") is False

    # Empty allowed_models means all models allowed
    store._engine._conn.executed = []
    store._engine._conn._rows = [
        {
            "id": "test-4",
            "name": "Model Key",
            "api_key_hash": store.hash_key(raw_key),
            "allowed_models": [],
            "enabled": True,
            "created_at": None,
            "updated_at": None,
        }
    ]
    assert await store.is_model_allowed(raw_key, "model-b") is True
