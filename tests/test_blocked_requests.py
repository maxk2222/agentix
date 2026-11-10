from __future__ import annotations

from typing import Any, List

import pytest

from app.metrics import BlockedRequestRecord, BlockedRequestStore
from app.security_middleware import RequestSizeLimitMiddleware


class FakeResult:
    def __init__(self, rows: List[dict[str, Any]]) -> None:
        self._rows = rows

    def mappings(self):
        return FakeMappings(self._rows)


class FakeMappings:
    def __init__(self, rows: List[dict[str, Any]]) -> None:
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return self._rows


class FakeConnection:
    def __init__(self, results: List[FakeResult] | None = None) -> None:
        self.results = results or []
        self.executed: List[tuple[str, dict[str, Any]]] = []

    async def execute(self, stmt, params=None):
        self.executed.append((str(stmt), params or {}))
        if self.results:
            return self.results.pop(0)
        return FakeResult([])

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None


class FakeEngine:
    def __init__(self, conn: FakeConnection) -> None:
        self._conn = conn

    def begin(self):
        return self._conn


class FakeApp:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, scope, receive, send):
        self.calls += 1
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


async def _collect_response(scope, middleware: RequestSizeLimitMiddleware):
    messages: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    await middleware(scope, receive, send)
    return messages


@pytest.mark.asyncio
async def test_blocked_ensure_tables():
    conn = FakeConnection()
    store = BlockedRequestStore(FakeEngine(conn))
    await store.ensure_tables()
    assert len(conn.executed) == 1
    assert "CREATE TABLE IF NOT EXISTS blocked_requests" in conn.executed[0][0]


@pytest.mark.asyncio
async def test_blocked_save():
    conn = FakeConnection()
    store = BlockedRequestStore(FakeEngine(conn))
    record = BlockedRequestRecord(
        path="/v1/chat/completions",
        method="POST",
        status_code=401,
        reason="Invalid API key",
    )
    await store.save(record)
    assert len(conn.executed) == 2
    assert "INSERT INTO blocked_requests" in conn.executed[1][0]


@pytest.mark.asyncio
async def test_blocked_get_summary():
    conn = FakeConnection(
        results=[
            FakeResult(
                [
                    {
                        "total_blocked": 15,
                        "auth_blocks": 10,
                        "size_blocks": 2,
                        "validation_blocks": 3,
                        "other_blocks": 0,
                    }
                ]
            )
        ]
    )
    store = BlockedRequestStore(FakeEngine(conn))
    store._tables_ensured = True
    summary = await store.get_summary(24)
    assert summary["total_blocked"] == 15
    assert summary["auth_blocks"] == 10
    assert summary["size_blocks"] == 2
    assert summary["validation_blocks"] == 3
    assert summary["other_blocks"] == 0


@pytest.mark.asyncio
async def test_blocked_get_history():
    conn = FakeConnection(
        results=[
            FakeResult(
                [
                    {
                        "bucket": "2024-01-01 00:00:00",
                        "blocked": 5,
                        "auth_blocks": 3,
                        "size_blocks": 1,
                        "validation_blocks": 1,
                    }
                ]
            )
        ]
    )
    store = BlockedRequestStore(FakeEngine(conn))
    store._tables_ensured = True
    rows = await store.get_history(24, "hour")
    assert len(rows) == 1
    assert rows[0]["bucket"] == "2024-01-01 00:00:00"
    assert rows[0]["blocked"] == 5
    assert rows[0]["auth_blocks"] == 3


@pytest.mark.asyncio
async def test_request_size_limit_exempts_admin_uploads():
    app = FakeApp()
    middleware = RequestSizeLimitMiddleware(
        app,
        max_bytes=10,
        exempt_paths={"/admin/knowledge/upload"},
    )

    messages = await _collect_response(
        {
            "type": "http",
            "path": "/admin/knowledge/upload",
            "method": "POST",
            "headers": [(b"content-length", b"999")],
        },
        middleware,
    )

    assert app.calls == 1
    assert any(message.get("status") == 200 for message in messages)


@pytest.mark.asyncio
async def test_request_size_limit_blocks_non_exempt_uploads():
    app = FakeApp()
    middleware = RequestSizeLimitMiddleware(app, max_bytes=10)

    messages = await _collect_response(
        {
            "type": "http",
            "path": "/v1/chat/completions",
            "method": "POST",
            "headers": [(b"content-length", b"999")],
        },
        middleware,
    )

    assert app.calls == 0
    assert any(message.get("status") == 413 for message in messages)
